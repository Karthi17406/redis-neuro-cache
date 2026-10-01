import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Chooses which keys to evict.
 *
 * The ML policy is the interesting one, but it is built
 * on top of LRU rather than beside it. Every ML decision
 * that cannot be made - service down, model missing,
 * request timed out, fewer victims returned than asked
 * for - falls through to the heuristic, so the cache
 * always gets an answer and never blocks on the network.
 *
 * The fallback is counted separately from the ML path,
 * because a policy that silently degrades to LRU while
 * reporting itself as ML would be indistinguishable from
 * one that works.
 */
public class EvictionManager {

    // =========================
    // CONFIGURATION
    // =========================

    private EvictionPolicy policy;

    private final MLClient mlClient;


    // =========================
    // COUNTERS
    // =========================

    private long evictionCount;

    private long mlDecisions;

    private long fallbackDecisions;


    // =========================
    // CONSTRUCTORS
    // =========================

    public EvictionManager() {

        this(defaultPolicy(), new MLClient());
    }


    public EvictionManager(
            EvictionPolicy policy,
            MLClient mlClient) {

        this.policy = policy;
        this.mlClient = mlClient;
    }


    /**
     * ML is opt in. A fresh database behaves exactly as
     * it did before this class existed unless the
     * operator asks for something else.
     */
    private static EvictionPolicy defaultPolicy() {

        String configured =
                System.getProperty("miniredis.eviction");

        if (configured == null) {
            return EvictionPolicy.LRU;
        }

        try {

            return EvictionPolicy.parse(configured);

        } catch (IllegalArgumentException e) {

            System.err.println(
                    "Unknown eviction policy '"
                            + configured
                            + "', using LRU. Choices: "
                            + EvictionPolicy.available());

            return EvictionPolicy.LRU;
        }
    }


    // =========================
    // ACCESSORS
    // =========================

    public synchronized EvictionPolicy getPolicy() {

        return policy;
    }


    public synchronized void setPolicy(EvictionPolicy policy) {

        this.policy = policy;
    }


    public MLClient getMlClient() {

        return mlClient;
    }


    public synchronized long getEvictionCount() {

        return evictionCount;
    }


    public synchronized long getMlDecisions() {

        return mlDecisions;
    }


    public synchronized long getFallbackDecisions() {

        return fallbackDecisions;
    }


    // =========================
    // SELECTION
    // =========================

    /**
     * Names the keys to drop, worst first.
     *
     * @param candidates key to feature vector, as the
     *                   tracker sees them right now
     * @param count      how many victims are needed
     * @return at most count keys; fewer only when there
     *         are not enough candidates
     */
    public synchronized List<String> selectVictims(
            Map<String, double[]> candidates,
            int count) {

        if (candidates == null
                || candidates.isEmpty()
                || count <= 0
                || policy == EvictionPolicy.NONE) {

            return List.of();
        }

        int wanted = Math.min(count, candidates.size());

        List<String> victims;

        if (policy == EvictionPolicy.ML) {

            victims = selectWithModel(candidates, wanted);

        } else {

            victims = selectWithHeuristic(
                    candidates, wanted, policy);

            fallbackDecisions++;
        }

        evictionCount += victims.size();

        return victims;
    }


    /**
     * Asks the model, and repairs a short or unusable
     * answer with the heuristic rather than evicting
     * fewer keys than the cache needs room for.
     */
    private List<String> selectWithModel(
            Map<String, double[]> candidates,
            int wanted) {

        List<String> ranked =
                mlClient.rankVictims(candidates, wanted);

        // Names the model invented, or keys that have
        // since gone, are not evictable.
        List<String> victims = new ArrayList<>();

        for (String key : ranked) {

            if (candidates.containsKey(key)
                    && !victims.contains(key)) {

                victims.add(key);
            }
        }

        if (victims.size() >= wanted) {

            mlDecisions++;

            return victims.subList(0, wanted);
        }

        // Either the service could not be reached or it
        // returned less than was asked for. Either way
        // the shortfall is filled by LRU.
        Map<String, double[]> remaining =
                new LinkedHashMap<>(candidates);

        victims.forEach(remaining::remove);

        victims.addAll(
                selectWithHeuristic(
                        remaining,
                        wanted - victims.size(),
                        EvictionPolicy.LRU));

        if (victims.isEmpty() || ranked.isEmpty()) {

            fallbackDecisions++;

        } else {

            mlDecisions++;
        }

        return victims;
    }


    /**
     * The classic policies, both expressed as a ranking
     * over the same feature vectors the model sees. LRU
     * drops the largest recency, LFU the smallest rate.
     */
    private List<String> selectWithHeuristic(
            Map<String, double[]> candidates,
            int wanted,
            EvictionPolicy heuristic) {

        if (wanted <= 0 || candidates.isEmpty()) {
            return List.of();
        }

        int index =
                heuristic == EvictionPolicy.LFU
                        ? MLFeatures.ACCESSES_PER_MINUTE
                        : MLFeatures.RECENCY_SECONDS;

        // LRU ranks worst-first by descending recency,
        // LFU by ascending access rate.
        boolean descending =
                heuristic != EvictionPolicy.LFU;

        List<Map.Entry<String, double[]>> entries =
                new ArrayList<>(candidates.entrySet());

        Comparator<Map.Entry<String, double[]>> byFeature =
                Comparator.comparingDouble(
                        entry -> value(entry.getValue(), index));

        entries.sort(
                descending ? byFeature.reversed() : byFeature);

        List<String> victims = new ArrayList<>();

        for (int i = 0;
                i < Math.min(wanted, entries.size());
                i++) {

            victims.add(entries.get(i).getKey());
        }

        return victims;
    }


    /**
     * A missing or malformed vector sorts as the most
     * evictable thing in the cache: a key we know
     * nothing about is the safest one to lose.
     */
    private static double value(double[] features, int index) {

        if (features == null || index >= features.length) {

            return Double.MAX_VALUE;
        }

        double raw = features[index];

        return Double.isFinite(raw) ? raw : Double.MAX_VALUE;
    }


    // =========================
    // REPORTING
    // =========================

    public synchronized String describe() {

        StringBuilder text = new StringBuilder();

        text.append("policy: ").append(policy);

        text.append("\nevictions: ").append(evictionCount);

        if (policy == EvictionPolicy.ML
                || mlDecisions > 0) {

            text.append("\nml decisions: ").append(mlDecisions);

            text.append("\nfallback decisions: ")
                .append(fallbackDecisions);

            text.append("\nml service: ")
                .append(mlClient.health());

            text.append("\nml calls: ")
                .append(mlClient.getSuccessCount())
                .append(" ok, ")
                .append(mlClient.getFailureCount())
                .append(" failed");

            String error = mlClient.getLastError();

            if (error != null) {

                text.append("\nlast ml error: ").append(error);
            }
        }

        return text.toString();
    }
}
