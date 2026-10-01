import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Keeps live access statistics for every key.
 *
 * This is the online half of the pipeline: the CSV log
 * feeds training, this tracker feeds prediction.
 */
public class KeyStatsTracker {

    private final Map<String, KeyStats> stats =
            new HashMap<>();


    // =========================
    // RECORD
    // =========================

    public synchronized void record(
            String key,
            String operation,
            int valueSize) {

        long now = System.currentTimeMillis();

        KeyStats keyStats = stats.get(key);

        if (keyStats == null) {

            keyStats = new KeyStats(now);

            stats.put(key, keyStats);
        }

        keyStats.record(operation, now, valueSize);
    }


    // =========================
    // FORGET
    // =========================

    /**
     * Drops the statistics of a key that no longer
     * exists, so the tracker cannot outgrow the
     * database itself.
     */
    public synchronized void forget(String key) {

        stats.remove(key);
    }


    public synchronized void clear() {

        stats.clear();
    }


    // =========================
    // LOOKUP
    // =========================

    public synchronized KeyStats get(String key) {

        return stats.get(key);
    }


    public synchronized boolean contains(String key) {

        return stats.containsKey(key);
    }


    public synchronized Set<String> trackedKeys() {

        return new HashMap<>(stats).keySet();
    }


    public synchronized int size() {

        return stats.size();
    }


    // =========================
    // FEATURES
    // =========================

    /**
     * Feature vector for one key, in MLFeatures order.
     * Returns null when the key was never seen.
     */
    public synchronized double[] featuresFor(String key) {

        KeyStats keyStats = stats.get(key);

        if (keyStats == null) {
            return null;
        }

        return keyStats.toFeatureVector(
                System.currentTimeMillis()
        );
    }


    /**
     * Feature vectors for many keys at once. Keys that
     * were never seen are skipped, so the caller reads
     * the returned map key set rather than assuming
     * the requested order came back intact.
     */
    public synchronized Map<String, double[]> featuresForAll(
            Iterable<String> keys) {

        Map<String, double[]> result =
                new HashMap<>();

        long now = System.currentTimeMillis();

        for (String key : keys) {

            KeyStats keyStats = stats.get(key);

            if (keyStats != null) {

                result.put(
                        key,
                        keyStats.toFeatureVector(now)
                );
            }
        }

        return result;
    }
}
