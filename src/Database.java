import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Database {

    // =========================
    // MAIN DATA
    // =========================

    // Main key-value database
    private HashMap<String, String> database =
            new HashMap<>();

    // Stores expiration time
    // key -> expiration timestamp in milliseconds
    private HashMap<String, Long> expiryTime =
            new HashMap<>();


    // =========================
    // PERSISTENCE
    // =========================

    private PersistenceManager persistenceManager =
            new PersistenceManager();


    // =========================
    // ACCESS STATISTICS
    // =========================

    // Offline: rows for the ML training pipeline
    private AccessLogger accessLogger =
            new AccessLogger();

    // Online: live features for eviction decisions
    private KeyStatsTracker statsTracker =
            new KeyStatsTracker();


    // =========================
    // CAPACITY
    // =========================

    // Zero means unlimited, which is how MiniRedis
    // behaved before eviction existed and is still the
    // default. Nothing is ever dropped unless an
    // operator asks for a bound.
    private int maxKeys = configuredMaxKeys();

    private EvictionManager evictionManager =
            new EvictionManager();


    private static int configuredMaxKeys() {

        String configured =
                System.getProperty("miniredis.maxkeys");

        if (configured == null) {
            return 0;
        }

        try {

            return Math.max(0, Integer.parseInt(configured.trim()));

        } catch (NumberFormatException e) {

            System.err.println(
                    "Ignoring bad miniredis.maxkeys: "
                            + configured);

            return 0;
        }
    }


    // =========================
    // CONSTRUCTOR
    // =========================

    @SuppressWarnings("unchecked")
    public Database() {

        // Load previously saved data
        HashMap<String, Object> loadedData =
                persistenceManager.loadData();

        if (!loadedData.isEmpty()) {

            database =
                    (HashMap<String, String>)
                            loadedData.get("database");

            expiryTime =
                    (HashMap<String, Long>)
                            loadedData.get("expiryTime");
        }
    }


    // =========================
    // SET
    // =========================

    public synchronized void set(
            String key,
            String value) {

        int valueSize =
                value == null ? 0 : value.length();

        boolean existed =
                database.containsKey(key);

        accessLogger.log(
                key,
                "SET",
                existed,
                valueSize
        );

        statsTracker.record(key, "SET", valueSize);

        // Store value
        database.put(key, value);

        // SET removes previous expiration
        expiryTime.remove(key);

        // Make room before persisting, so what is saved
        // is what the database actually holds.
        enforceCapacity(key);

        // Save data
        persistenceManager.saveData(
                database,
                expiryTime
        );
    }


    // =========================
    // GET
    // =========================

    public synchronized String get(String key) {

        return lookup(key, "GET");
    }


    /**
     * Shared read path for GET and EXISTS.
     *
     * Both need the same expiry handling, but each has
     * to appear in the access log under its own name,
     * otherwise EXISTS would inflate the GET counts the
     * model trains on.
     */
    private synchronized String lookup(
            String key,
            String operation) {

        // Check if key has an expiration
        if (expiryTime.containsKey(key)) {

            long expiration =
                    expiryTime.get(key);

            // Key has expired
            if (System.currentTimeMillis()
                    >= expiration) {

                database.remove(key);
                expiryTime.remove(key);

                statsTracker.forget(key);

                // Save updated data
                persistenceManager.saveData(
                        database,
                        expiryTime
                );

                accessLogger.log(
                        key,
                        operation,
                        false,
                        -1
                );

                return null;
            }
        }

        String value =
                database.get(key);

        int valueSize =
                value == null ? -1 : value.length();

        accessLogger.log(
                key,
                operation,
                value != null,
                valueSize
        );

        // A miss must not create statistics for a key
        // that was never stored, or the tracker would
        // fill up with typos and probe traffic.
        if (value != null) {

            statsTracker.record(
                    key,
                    operation,
                    valueSize
            );
        }

        return value;
    }


    // =========================
    // DELETE
    // =========================

    public synchronized boolean delete(String key) {

        boolean removed =
                database.remove(key) != null;

        accessLogger.log(
                key,
                "DELETE",
                removed,
                -1
        );

        // Remove expiration information too
        expiryTime.remove(key);

        statsTracker.forget(key);

        if (removed) {

            persistenceManager.saveData(
                    database,
                    expiryTime
            );
        }

        return removed;
    }


    // =========================
    // EXISTS
    // =========================

    public synchronized boolean exists(String key) {

        return lookup(key, "EXISTS") != null;
    }


    // =========================
    // KEYS
    // =========================

    public synchronized Set<String> keys() {

        boolean changed = false;

        // Remove expired keys first
        for (String key :
                new HashMap<>(expiryTime).keySet()) {

            if (System.currentTimeMillis()
                    >= expiryTime.get(key)) {

                database.remove(key);
                expiryTime.remove(key);

                statsTracker.forget(key);

                changed = true;
            }
        }

        // Save if expired keys were removed
        if (changed) {

            persistenceManager.saveData(
                    database,
                    expiryTime
            );
        }

        return database.keySet();
    }


    // =========================
    // FLUSH
    // =========================

    public synchronized void flush() {

        database.clear();
        expiryTime.clear();

        statsTracker.clear();

        // Save empty database
        persistenceManager.saveData(
                database,
                expiryTime
        );
    }


    // =========================
    // EXPIRE
    // =========================

    public synchronized boolean expire(
            String key,
            long seconds) {

        // Key must exist
        if (!database.containsKey(key)) {

            return false;
        }

        long expiration =
                System.currentTimeMillis()
                + (seconds * 1000);

        expiryTime.put(
                key,
                expiration
        );

        // Save expiration
        persistenceManager.saveData(
                database,
                expiryTime
        );

        return true;
    }


    // =========================
    // TTL
    // =========================

    public synchronized long ttl(String key) {

        // Key doesn't exist
        if (!database.containsKey(key)) {

            return -2;
        }

        // Key exists but has no expiration
        if (!expiryTime.containsKey(key)) {

            return -1;
        }

        long remaining =
                expiryTime.get(key)
                - System.currentTimeMillis();

        // Key has expired
        if (remaining <= 0) {

            database.remove(key);
            expiryTime.remove(key);

            statsTracker.forget(key);

            persistenceManager.saveData(
                    database,
                    expiryTime
            );

            return -2;
        }

        return remaining / 1000;
    }


    // =========================
    // EVICTION
    // =========================

    /**
     * Drops keys until the database fits its bound.
     *
     * Only ever called from a write, because that is the
     * only thing that can make the database too big.
     * Reads stay free of network calls no matter which
     * policy is configured.
     */
    private synchronized int enforceCapacity(String justWritten) {

        if (maxKeys <= 0
                || database.size() <= maxKeys) {

            return 0;
        }

        // NONE is an instruction, not a policy that
        // happens to pick nothing. The bound is allowed
        // to be exceeded rather than enforced behind the
        // operator back.
        if (evictionManager.getPolicy()
                == EvictionPolicy.NONE) {

            return 0;
        }

        int excess = database.size() - maxKeys;

        Map<String, double[]> candidates =
                statsTracker.featuresForAll(database.keySet());

        // The key that caused the overflow is not a
        // candidate to resolve it. Evicting it would
        // make the write that triggered this a silent
        // no-op, and under LFU it would happen every
        // time, since a key written once is always the
        // least frequently used thing in the cache.
        if (justWritten != null) {

            candidates.remove(justWritten);
        }

        List<String> victims =
                evictionManager.selectVictims(candidates, excess);

        int removed = 0;

        for (String victim : victims) {

            if (evict(victim)) {
                removed++;
            }
        }

        // A policy that returned nothing usable must not
        // leave the database over its bound, so recency
        // decides the rest.
        if (removed < excess) {

            for (String key : lruOrder()) {

                if (removed >= excess) {
                    break;
                }

                if (key.equals(justWritten)) {
                    continue;
                }

                if (evict(key)) {
                    removed++;
                }
            }
        }

        return removed;
    }


    /**
     * Removes one key as an eviction rather than as a
     * client DELETE.
     *
     * The operation is logged under its own name so the
     * training pipeline can tell a key the cache threw
     * away apart from one a client asked to remove. Both
     * forget the key, but only DELETE means the client
     * no longer wants it.
     */
    private synchronized boolean evict(String key) {

        if (database.remove(key) == null) {

            return false;
        }

        expiryTime.remove(key);

        accessLogger.log(key, "EVICT", true, -1);

        statsTracker.forget(key);

        return true;
    }


    /**
     * Least recently used first, straight from the
     * tracker. This is the last resort inside
     * enforceCapacity, so it must not depend on the
     * eviction manager or on the model.
     */
    private synchronized List<String> lruOrder() {

        List<String> keys =
                new ArrayList<>(database.keySet());

        keys.sort(
                Comparator.comparingLong(
                        key -> {

                            KeyStats stats =
                                    statsTracker.get(key);

                            // An untracked key is the
                            // safest thing to lose.
                            return stats == null
                                    ? Long.MIN_VALUE
                                    : stats.getLastSeenMillis();
                        }));

        return keys;
    }


    public synchronized int getMaxKeys() {

        return maxKeys;
    }


    /**
     * @return how many keys had to be evicted to honour
     *         the new bound
     */
    public synchronized int setMaxKeys(int maxKeys) {

        this.maxKeys = Math.max(0, maxKeys);

        int removed = enforceCapacity(null);

        if (removed > 0) {

            persistenceManager.saveData(
                    database,
                    expiryTime
            );
        }

        return removed;
    }


    public synchronized EvictionManager getEvictionManager() {

        return evictionManager;
    }


    public synchronized void setEvictionManager(
            EvictionManager evictionManager) {

        this.evictionManager = evictionManager;
    }


    // =========================
    // STATISTICS ACCESS
    // =========================

    public synchronized KeyStatsTracker getStatsTracker() {

        return statsTracker;
    }


    public synchronized int size() {

        return database.size();
    }
}