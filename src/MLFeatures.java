/**
 * The feature contract between Java and Python.
 *
 * Both sides build feature vectors in this exact order.
 * The Python trainer asserts the same list, so a change
 * on one side that is not mirrored on the other gets
 * caught instead of silently scoring garbage.
 *
 * Every feature here has to be computable from the live
 * KeyStatsTracker, otherwise the model could be trained
 * on something that is not available at eviction time.
 * That rules out delete_count, for example: DELETE drops
 * the key from the tracker, so at prediction time the
 * count is always zero no matter what the log says.
 */
public final class MLFeatures {

    public static final String[] NAMES = {
            "total_accesses",
            "get_count",
            "set_count",
            "age_seconds",
            "recency_seconds",
            "mean_interval_seconds",
            "ewma_interval_seconds",
            "accesses_per_minute",
            "get_ratio",
            "value_size",
            "recency_over_mean_interval"
    };

    public static int count() {

        return NAMES.length;
    }


    /**
     * Position of a feature in a vector.
     *
     * The heuristic eviction policies need to read one
     * particular feature out of a vector. Looking the
     * index up by name keeps them correct if the list is
     * ever reordered.
     */
    public static int indexOf(String name) {

        for (int i = 0; i < NAMES.length; i++) {

            if (NAMES[i].equals(name)) {
                return i;
            }
        }

        throw new IllegalArgumentException(
                "Unknown feature: " + name);
    }


    // Indices the eviction policies rank on.
    public static final int RECENCY_SECONDS =
            indexOf("recency_seconds");

    public static final int ACCESSES_PER_MINUTE =
            indexOf("accesses_per_minute");

    private MLFeatures() {
        // Constants only
    }
}
