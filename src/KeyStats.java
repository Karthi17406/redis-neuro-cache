/**
 * Access statistics for a single key.
 *
 * The same numbers are rebuilt offline from the access
 * log by the Python feature builder, so the model sees
 * the same features at training time and at eviction
 * time.
 *
 * Anything added here has to be mirrored in
 * ml/features.py, and the order has to match
 * MLFeatures.NAMES.
 */
public class KeyStats {

    // =========================
    // CONFIGURATION
    // =========================

    // Weight of the newest gap in the EWMA.
    // Higher values react faster to recent changes.
    public static final double EWMA_ALPHA = 0.3;


    // =========================
    // COUNTERS
    // =========================

    private long totalAccesses;

    private long getCount;

    private long setCount;

    private long deleteCount;


    // =========================
    // TIMESTAMPS
    // =========================

    private long firstSeenMillis;

    private long lastSeenMillis;


    // Sum of gaps between consecutive accesses.
    // Used to calculate the mean interval.
    private long intervalSumMillis;

    private long intervalCount;

    // Exponentially weighted moving average of access gaps.
    private double ewmaIntervalMillis;


    // =========================
    // VALUE
    // =========================

    private int valueSize;


    // =========================
    // CONSTRUCTOR
    // =========================

    public KeyStats(long nowMillis) {

        this.firstSeenMillis = nowMillis;
        this.lastSeenMillis = nowMillis;
    }


    // =========================
    // RECORD ACCESS
    // =========================

    public synchronized void record(
            String operation,
            long nowMillis,
            int valueSize) {

        // Calculate the gap from the previous access.
        if (totalAccesses > 0) {

            long gap = nowMillis - lastSeenMillis;

            if (gap >= 0) {

                intervalSumMillis += gap;
                intervalCount++;

                // First measured interval becomes
                // the initial EWMA value.
                if (intervalCount == 1) {

                    ewmaIntervalMillis = gap;

                } else {

                    ewmaIntervalMillis =
                            EWMA_ALPHA * gap
                            + (1.0 - EWMA_ALPHA)
                            * ewmaIntervalMillis;
                }
            }
        }

        // Count this access.
        totalAccesses++;

        // Update last access time.
        lastSeenMillis = nowMillis;


        // Count the operation type.
        switch (operation) {

            case "GET":
                getCount++;
                break;

            case "SET":
                setCount++;
                break;

            case "DELETE":
                deleteCount++;
                break;

            default:
                // EXISTS and other operations still
                // count as accesses.
                break;
        }


        // Update the value size when a valid size
        // is provided.
        if (valueSize >= 0) {

            this.valueSize = valueSize;
        }
    }


    // =========================
    // FEATURE VECTOR
    // =========================

    /**
     * Builds the feature vector in the canonical order
     * defined by MLFeatures.NAMES.
     */
    public synchronized double[] toFeatureVector(
            long nowMillis) {

        // How long the key has existed.
        double ageSeconds =
                Math.max(
                        0.0,
                        (nowMillis - firstSeenMillis)
                                / 1000.0
                );


        // How long since the key was last accessed.
        double recencySeconds =
                Math.max(
                        0.0,
                        (nowMillis - lastSeenMillis)
                                / 1000.0
                );


        // Average time between accesses.
        //
        // If there is only one access, there is no
        // measured interval yet, so use the key age.
        double meanIntervalSeconds =
                intervalCount > 0
                        ? (intervalSumMillis
                                / (double) intervalCount)
                                / 1000.0
                        : ageSeconds;


        // EWMA version of the access interval.
        double ewmaIntervalSeconds =
                intervalCount > 0
                        ? ewmaIntervalMillis / 1000.0
                        : ageSeconds;


        // Number of accesses per minute.
        //
        // The minimum denominator prevents division by
        // zero or extremely large values for brand-new keys.
        double accessesPerMinute =
                totalAccesses
                        / Math.max(
                                ageSeconds / 60.0,
                                1.0 / 60.0
                        );


        // Percentage of accesses that were GET operations.
        double getRatio =
                totalAccesses > 0
                        ? getCount / (double) totalAccesses
                        : 0.0;


        // How overdue the key currently is.
        //
        // > 1 means the key has been inactive for longer
        // than its usual average interval.
        double recencyOverMeanInterval =
                recencySeconds
                        / Math.max(
                                meanIntervalSeconds,
                                0.001
                        );


        // IMPORTANT:
        // The order here must match MLFeatures.NAMES.
        return new double[]{
                totalAccesses,
                getCount,
                setCount,
                ageSeconds,
                recencySeconds,
                meanIntervalSeconds,
                ewmaIntervalSeconds,
                accessesPerMinute,
                getRatio,
                valueSize,
                recencyOverMeanInterval
        };
    }


    // =========================
    // GETTERS
    // =========================

    public synchronized long getTotalAccesses() {

        return totalAccesses;
    }


    public synchronized long getGetCount() {

        return getCount;
    }


    public synchronized long getSetCount() {

        return setCount;
    }


    public synchronized long getDeleteCount() {

        return deleteCount;
    }


    public synchronized long getLastSeenMillis() {

        return lastSeenMillis;
    }


    public synchronized long getFirstSeenMillis() {

        return firstSeenMillis;
    }


    public synchronized int getValueSize() {

        return valueSize;
    }
}