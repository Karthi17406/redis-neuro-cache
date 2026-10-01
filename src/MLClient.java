import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Talks to the Python model service.
 *
 * The cache must keep working when the model does not.
 * If the model service is down, slow, untrained, or
 * returning invalid data, MiniRedis falls back to its
 * normal eviction behavior instead of breaking.
 *
 * A short circuit breaker prevents a dead model service
 * from causing a timeout on every eviction attempt.
 *
 * Optional configuration:
 *
 * -Dminiredis.ml.url=http://127.0.0.1:8000
 * -Dminiredis.ml.enabled=false
 * -Dminiredis.ml.timeoutms=250
 */
public class MLClient {

    // =========================
    // CONFIGURATION
    // =========================

    private static final String DEFAULT_URL =
            "http://127.0.0.1:8000";

    private static final long DEFAULT_TIMEOUT_MS = 250;

    // Stop calling the service for this amount of time
    // after a failure.
    private static final long CIRCUIT_OPEN_MS = 5000;


    private final String baseUrl;

    private final Duration timeout;

    private final boolean enabled;


    // Created only when the first HTTP request is made.
    private volatile HttpClient httpClient;


    // =========================
    // CIRCUIT BREAKER STATE
    // =========================

    private volatile long circuitOpenUntil;

    private volatile String lastError;

    private volatile long successCount;

    private volatile long failureCount;


    // =========================
    // CONSTRUCTORS
    // =========================

    /**
     * Creates an MLClient using system-property
     * configuration.
     */
    public MLClient() {

        this(
                System.getProperty(
                        "miniredis.ml.url",
                        DEFAULT_URL
                ),

                parseTimeout(),

                !"false".equalsIgnoreCase(
                        System.getProperty(
                                "miniredis.ml.enabled",
                                "true"
                        )
                )
        );
    }


    /**
     * Creates an MLClient with explicit configuration.
     */
    public MLClient(
            String baseUrl,
            long timeoutMillis,
            boolean enabled) {

        // Remove trailing slashes so that:
        //
        // baseUrl + "/health"
        //
        // does not become:
        //
        // http://localhost:8000//health
        while (baseUrl.endsWith("/")) {

            baseUrl = baseUrl.substring(
                    0,
                    baseUrl.length() - 1
            );
        }

        this.baseUrl = baseUrl;

        this.timeout =
                Duration.ofMillis(
                        Math.max(1, timeoutMillis)
                );

        this.enabled = enabled;
    }


    /**
     * Reads the timeout safely from the system property.
     *
     * If the value is invalid, the default timeout is used.
     */
    private static long parseTimeout() {

        String value =
                System.getProperty(
                        "miniredis.ml.timeoutms",
                        String.valueOf(DEFAULT_TIMEOUT_MS)
                );

        try {

            return Long.parseLong(value);

        } catch (NumberFormatException e) {

            return DEFAULT_TIMEOUT_MS;
        }
    }


    /**
     * Creates the HttpClient lazily.
     *
     * This avoids creating HTTP resources when ML is
     * disabled or when no request is ever made.
     */
    private HttpClient httpClient() {

        HttpClient client = httpClient;

        if (client != null) {
            return client;
        }

        synchronized (this) {

            if (httpClient == null) {

                httpClient =
                        HttpClient.newBuilder()
                                .connectTimeout(timeout)
                                .version(
                                        HttpClient.Version.HTTP_1_1
                                )
                                .build();
            }

            return httpClient;
        }
    }


    // =========================
    // STATUS
    // =========================

    public boolean isEnabled() {

        return enabled;
    }


    /**
     * Returns true when a request would actually
     * be attempted.
     */
    public boolean isCircuitClosed() {

        return enabled
                && System.currentTimeMillis()
                        >= circuitOpenUntil;
    }


    public String getLastError() {

        return lastError;
    }


    public long getSuccessCount() {

        return successCount;
    }


    public long getFailureCount() {

        return failureCount;
    }


    public String getBaseUrl() {

        return baseUrl;
    }


    /**
     * Checks whether the Python model service is reachable
     * and whether a model is loaded.
     */
    public String health() {

        if (!enabled) {

            return "disabled";
        }

        try {

            HttpResponse<String> response =
                    send("/health", null);

            if (response.statusCode() != 200) {

                return "http "
                        + response.statusCode();
            }

            Map<String, Object> body =
                    Json.parseObject(
                            response.body()
                    );

            boolean loaded =
                    Boolean.TRUE.equals(
                            body.get("model_loaded")
                    );

            if (!loaded) {

                return "reachable, no model loaded";
            }

            return "ready ("
                    + body.get("model_name")
                    + ")";

        } catch (Exception e) {

            return "unreachable: "
                    + e.getMessage();
        }
    }


    // =========================
    // EVICTION RANKING
    // =========================

    /**
     * Asks the model which keys should be evicted.
     *
     * @param candidates key -> feature vector
     * @param count number of victims requested
     *
     * @return victims worst first, or an empty list
     *         when the model cannot be consulted
     */
    public List<String> rankVictims(
            Map<String, double[]> candidates,
            int count) {

        if (candidates == null
                || candidates.isEmpty()
                || count <= 0) {

            return List.of();
        }

        if (!isCircuitClosed()) {

            return List.of();
        }

        try {

            String body =
                    buildEvictRequest(
                            candidates,
                            count
                    );

            HttpResponse<String> response =
                    send("/evict", body);

            if (response.statusCode() != 200) {

                recordFailure(
                        "http "
                                + response.statusCode()
                                + ": "
                                + truncate(
                                        response.body()
                                )
                );

                return List.of();
            }

            Map<String, Object> parsed =
                    Json.parseObject(
                            response.body()
                    );

            Object victims =
                    parsed.get("victims");


            if (!(victims instanceof List<?>)) {

                recordFailure(
                        "response had no victims array"
                );

                return List.of();
            }


            List<?> list =
                    (List<?>) victims;

            List<String> result =
                    new ArrayList<>();


            for (Object victim : list) {

                if (victim instanceof String) {

                    result.add(
                            (String) victim
                    );
                }
            }


            recordSuccess();

            return result;

        } catch (Exception e) {

            recordFailure(
                    e.getClass().getSimpleName()
                            + ": "
                            + e.getMessage()
            );

            return List.of();
        }
    }


    // =========================
    // PREDICTION
    // =========================

    /**
     * Requests reuse probabilities from the ML service.
     *
     * @return key -> reuse probability
     *
     *         Returns an empty map if the model could
     *         not be consulted.
     */
    public Map<String, Double> predict(
            Map<String, double[]> candidates) {

        if (candidates == null
                || candidates.isEmpty()) {

            return Map.of();
        }

        if (!isCircuitClosed()) {

            return Map.of();
        }

        try {

            String body =
                    buildPredictRequest(
                            candidates
                    );

            HttpResponse<String> response =
                    send("/predict", body);


            if (response.statusCode() != 200) {

                recordFailure(
                        "http "
                                + response.statusCode()
                                + ": "
                                + truncate(
                                        response.body()
                                )
                );

                return Map.of();
            }


            Map<String, Object> parsed =
                    Json.parseObject(
                            response.body()
                    );


            Object predictions =
                    parsed.get("predictions");


            if (!(predictions instanceof List<?>)) {

                recordFailure(
                        "response had no predictions array"
                );

                return Map.of();
            }


            List<?> list =
                    (List<?>) predictions;


            Map<String, Double> result =
                    new LinkedHashMap<>();


            for (Object item : list) {

                if (!(item instanceof Map<?, ?>)) {

                    continue;
                }


                Map<?, ?> entry =
                        (Map<?, ?>) item;


                Object key =
                        entry.get("key");


                Object probability =
                        entry.get(
                                "reuse_probability"
                        );


                if (key instanceof String
                        && probability instanceof Double) {

                    result.put(
                            (String) key,
                            (Double) probability
                    );
                }
            }


            recordSuccess();

            return result;

        } catch (Exception e) {

            recordFailure(
                    e.getClass().getSimpleName()
                            + ": "
                            + e.getMessage()
            );

            return Map.of();
        }
    }


    // =========================
    // REQUEST BUILDING
    // =========================

    /**
     * Converts candidates into JSON.
     *
     * Features are sent using their names rather than
     * as a bare array.
     *
     * This makes it easier for the Python service to
     * detect feature-order mismatches.
     */
    private String candidatesJson(
            Map<String, double[]> candidates) {

        StringBuilder json =
                new StringBuilder();

        json.append("[");

        boolean firstCandidate = true;


        for (Map.Entry<String, double[]> entry
                : candidates.entrySet()) {

            double[] features =
                    entry.getValue();


            // Only send vectors that have exactly the
            // expected number of features.
            if (features == null
                    || features.length
                            != MLFeatures.NAMES.length) {

                continue;
            }


            if (!firstCandidate) {

                json.append(",");
            }

            firstCandidate = false;


            json.append("{\"key\":")
                    .append(
                            Json.quote(
                                    entry.getKey()
                            )
                    )
                    .append(",\"features\":{");


            for (int i = 0;
                    i < features.length;
                    i++) {

                if (i > 0) {

                    json.append(",");
                }


                json.append(
                                Json.quote(
                                        MLFeatures.NAMES[i]
                                )
                        )
                        .append(":")
                        .append(
                                Json.number(
                                        features[i]
                                )
                        );
            }


            json.append("}}");
        }


        json.append("]");

        return json.toString();
    }


    /**
     * Builds the /evict request body.
     */
    private String buildEvictRequest(
            Map<String, double[]> candidates,
            int count) {

        return "{\"count\":"
                + count
                + ",\"candidates\":"
                + candidatesJson(candidates)
                + "}";
    }


    /**
     * Builds the /predict request body.
     */
    private String buildPredictRequest(
            Map<String, double[]> candidates) {

        return "{\"candidates\":"
                + candidatesJson(candidates)
                + "}";
    }


    // =========================
    // HTTP
    // =========================

    /**
     * Sends either a GET or POST request.
     */
    private HttpResponse<String> send(
            String path,
            String body) throws Exception {

        HttpRequest.Builder builder =
                HttpRequest.newBuilder()
                        .uri(
                                URI.create(
                                        baseUrl + path
                                )
                        )
                        .timeout(timeout);


        if (body == null) {

            builder.GET();

        } else {

            builder.header(
                            "Content-Type",
                            "application/json"
                    )
                    .POST(
                            HttpRequest.BodyPublishers
                                    .ofString(body)
                    );
        }


        return httpClient().send(
                builder.build(),
                HttpResponse.BodyHandlers.ofString()
        );
    }


    // =========================
    // CIRCUIT BREAKER
    // =========================

    /**
     * Called when the model request succeeds.
     */
    private synchronized void recordSuccess() {

        successCount++;

        lastError = null;

        circuitOpenUntil = 0;
    }


    /**
     * Called when the model request fails.
     *
     * Opens the circuit for a short period so repeated
     * eviction operations do not repeatedly wait for
     * a dead model service.
     */
    private synchronized void recordFailure(
            String message) {

        failureCount++;

        lastError = message;

        circuitOpenUntil =
                System.currentTimeMillis()
                        + CIRCUIT_OPEN_MS;
    }


    /**
     * Prevents very large HTTP error responses from
     * being stored in lastError.
     */
    private static String truncate(String text) {

        if (text == null) {

            return "";
        }

        return text.length() <= 200
                ? text
                : text.substring(0, 200) + "...";
    }
}