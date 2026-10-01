import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates a realistic access workload through the
 * real Database, so the access log is produced by the
 * same code path that runs in production.
 *
 * The original AccessLogGenerator is kept as is. It
 * produces a handful of keys with fixed frequencies,
 * which is enough to demo the log but far too little
 * signal to train on: every key is either always hot
 * or always cold, so frequency alone separates them
 * perfectly and the model learns nothing interesting.
 *
 * This generator adds the properties a cache model
 * actually has to cope with:
 *
 *   - a skewed key popularity distribution
 *   - keys that go hot for a while and then cool down
 *   - one shot keys that are never touched again
 *   - churn, so keys get created and deleted mid run
 *
 * Usage:
 *   java WorkloadGenerator [operations] [seed]
 */
public class WorkloadGenerator {

    // =========================
    // KEY CLASSES
    // =========================

    private static final int HOT_KEYS = 12;

    private static final int WARM_KEYS = 40;

    private static final int COLD_KEYS = 90;

    private static final int ONESHOT_KEYS = 60;

    // How many phases the run is split into. A new set
    // of warm keys gets promoted in every phase.
    private static final int PHASES = 6;


    public static void main(String[] args) {

        int operations =
                args.length > 0
                        ? Integer.parseInt(args[0])
                        : 12000;

        long seed =
                args.length > 1
                        ? Long.parseLong(args[1])
                        : 42L;

        Random random = new Random(seed);

        Database database = new Database();

        // =========================
        // BUILD KEY POPULATION
        // =========================

        List<String> hot = new ArrayList<>();
        List<String> warm = new ArrayList<>();
        List<String> cold = new ArrayList<>();
        List<String> oneshot = new ArrayList<>();

        for (int i = 0; i < HOT_KEYS; i++) {
            String key = "hot:" + i;
            hot.add(key);
            database.set(key, payload(random, 64));
        }

        for (int i = 0; i < WARM_KEYS; i++) {
            String key = "warm:" + i;
            warm.add(key);
            database.set(key, payload(random, 32));
        }

        for (int i = 0; i < COLD_KEYS; i++) {
            String key = "cold:" + i;
            cold.add(key);
            database.set(key, payload(random, 16));
        }

        for (int i = 0; i < ONESHOT_KEYS; i++) {
            oneshot.add("session:" + i);
        }

        System.out.println(
                "Key population: "
                        + (hot.size() + warm.size()
                           + cold.size() + oneshot.size())
                        + " keys"
        );

        // =========================
        // MAIN LOOP
        // =========================

        int perPhase =
                Math.max(1, operations / PHASES);

        // Warm keys promoted to hot for the phase
        List<String> promoted = new ArrayList<>();

        int oneshotCursor = 0;

        long start = System.currentTimeMillis();

        for (int op = 0; op < operations; op++) {

            // Start of a new phase: re promote
            if (op % perPhase == 0) {

                promoted.clear();

                for (int i = 0; i < 6; i++) {

                    promoted.add(
                            warm.get(
                                    random.nextInt(
                                            warm.size())));
                }

                System.out.println(
                        "Phase "
                                + (op / perPhase + 1)
                                + " promoted: "
                                + promoted
                );
            }

            double roll = random.nextDouble();

            String key;

            if (roll < 0.45) {

                // Always hot keys
                key = hot.get(
                        random.nextInt(hot.size()));

            } else if (roll < 0.70) {

                // Temporarily hot keys
                key = promoted.get(
                        random.nextInt(promoted.size()));

            } else if (roll < 0.88) {

                key = warm.get(
                        random.nextInt(warm.size()));

            } else if (roll < 0.97) {

                key = cold.get(
                        random.nextInt(cold.size()));

            } else {

                // One shot key: created, read once,
                // then abandoned forever
                if (oneshotCursor < oneshot.size()) {

                    key = oneshot.get(oneshotCursor++);

                    database.set(
                            key,
                            payload(random, 24)
                    );

                } else {

                    key = cold.get(
                            random.nextInt(cold.size()));
                }
            }

            // =========================
            // OPERATION MIX
            // =========================

            double opRoll = random.nextDouble();

            if (opRoll < 0.88) {

                database.get(key);

            } else if (opRoll < 0.97) {

                database.set(
                        key,
                        payload(random, 32)
                );

            } else if (opRoll < 0.99) {

                database.exists(key);

            } else {

                // Churn: delete a cold key and recreate
                // it, so the log contains real deletes
                String victim =
                        cold.get(
                                random.nextInt(
                                        cold.size()));

                database.delete(victim);

                database.set(
                        victim,
                        payload(random, 16)
                );
            }

            // Spread the run over real wall clock time
            // so the recency and age features vary.
            if (op % 20 == 0) {

                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }

        // Buffered rows must reach the disk before the
        // Python side reads the file.
        AccessLogger.flush();

        long elapsed =
                System.currentTimeMillis() - start;

        System.out.println(
                "\nWorkload complete: "
                        + operations
                        + " operations in "
                        + elapsed
                        + " ms"
        );

        System.out.println(
                "Distinct keys tracked: "
                        + database.getStatsTracker().size()
        );
    }


    // =========================
    // PAYLOAD
    // =========================

    /**
     * Builds a value of a randomised length, so the
     * value_size feature is not a constant.
     */
    private static String payload(
            Random random,
            int maxLength) {

        int length =
                4 + random.nextInt(
                        Math.max(1, maxLength));

        StringBuilder builder =
                new StringBuilder(length);

        for (int i = 0; i < length; i++) {

            builder.append(
                    (char) ('a' + random.nextInt(26)));
        }

        return builder.toString();
    }
}
