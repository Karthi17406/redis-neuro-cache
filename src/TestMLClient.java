import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tests the JSON codec and the ML bridge.
 *
 * The degradation tests matter more than the happy-path
 * tests. The model service is optional, so MiniRedis must
 * continue working when the service is missing, slow,
 * disabled, or unavailable.
 *
 * Run with the Python service up to exercise the live
 * calls as well. Without it, the live section is skipped.
 */
public class TestMLClient {

    private static int passed;

    private static int failed;


    public static void main(String[] args) {

        System.out.println(
                "Testing JSON codec and ML client\n"
        );

        testJsonWriting();

        testJsonParsing();

        testJsonErrors();

        testClientDegradation();

        testRequestShape();

        testLiveService();


        System.out.println(
                "\n"
                        + passed
                        + " passed, "
                        + failed
                        + " failed"
        );


        if (failed > 0) {

            System.exit(1);
        }
    }


    // =========================
    // JSON WRITING
    // =========================

    private static void testJsonWriting() {

        section("JSON writing");


        check(
                "plain string",
                Json.quote("hello")
                        .equals("\"hello\"")
        );


        check(
                "quotes escaped",
                Json.quote("say \"hi\"")
                        .equals("\"say \\\"hi\\\"\"")
        );


        check(
                "backslash escaped",
                Json.quote("a\\b")
                        .equals("\"a\\\\b\"")
        );


        check(
                "newline escaped",
                Json.quote("a\nb")
                        .equals("\"a\\nb\"")
        );


        check(
                "control character escaped",
                Json.quote("a\u0001b")
                        .equals("\"a\\u0001b\"")
        );


        check(
                "finite number",
                Json.number(1.5)
                        .equals("1.5")
        );


        check(
                "NaN becomes null",
                Json.number(Double.NaN)
                        .equals("null")
        );


        check(
                "infinity becomes null",
                Json.number(
                        Double.POSITIVE_INFINITY
                ).equals("null")
        );
    }


    // =========================
    // JSON PARSING
    // =========================

    private static void testJsonParsing() {

        section("JSON parsing");


        Map<String, Object> flat =
                Json.parseObject(
                        "{\"a\":1,\"b\":\"two\","
                                + "\"c\":true,\"d\":null}"
                );


        check(
                "number parsed",
                Double.valueOf(1.0)
                        .equals(flat.get("a"))
        );


        check(
                "string parsed",
                "two".equals(flat.get("b"))
        );


        check(
                "boolean parsed",
                Boolean.TRUE.equals(
                        flat.get("c")
                )
        );


        check(
                "null parsed",
                flat.containsKey("d")
                        && flat.get("d") == null
        );


        // Exact shape returned by /evict.
        Map<String, Object> response =
                Json.parseObject(
                        "{\"model_name\":\"random_forest\","
                                + "\"victims\":[\"cold:3\","
                                + "\"session:9\"],"
                                + "\"scores\":["
                                + "{\"key\":\"cold:3\","
                                + "\"reuse_probability\":0.0125}]}"
                );


        check(
                "victims array parsed",
                isVictimsArrayCorrect(response)
        );


        check(
                "nested object parsed",
                isNestedObjectCorrect(response)
        );


        check(
                "exponent parsed",
                Double.valueOf(1.2e-3)
                        .equals(
                                Json.parseObject(
                                        "{\"x\":1.2e-3}"
                                ).get("x")
                        )
        );


        check(
                "negative parsed",
                Double.valueOf(-4.0)
                        .equals(
                                Json.parseObject(
                                        "{\"x\":-4}"
                                ).get("x")
                        )
        );


        check(
                "unicode escape parsed",
                "\u00e9".equals(
                        Json.parseObject(
                                "{\"x\":\"\\u00e9\"}"
                        ).get("x")
                )
        );


        check(
                "whitespace tolerated",
                Json.parseObject(
                        "  { \"x\" : 1 }  "
                ).containsKey("x")
        );


        check(
                "empty object",
                Json.parseObject("{}").isEmpty()
        );


        check(
                "empty array",
                isEmptyArray(
                        Json.parse("[]")
                )
        );


        // A key containing quotes, backslashes and
        // newline characters must survive a round trip.
        String awkward =
                "key\"with\\odd\nchars";


        check(
                "round trip",
                awkward.equals(
                        Json.parse(
                                Json.quote(awkward)
                        )
                )
        );
    }


    private static boolean isVictimsArrayCorrect(
            Map<String, Object> response) {

        Object value =
                response.get("victims");


        if (!(value instanceof List<?>)) {

            return false;
        }


        List<?> list =
                (List<?>) value;


        return list.size() == 2
                && "cold:3".equals(list.get(0));
    }


    private static boolean isNestedObjectCorrect(
            Map<String, Object> response) {

        Object value =
                response.get("scores");


        if (!(value instanceof List<?>)) {

            return false;
        }


        List<?> list =
                (List<?>) value;


        if (list.isEmpty()) {

            return false;
        }


        Object first =
                list.get(0);


        if (!(first instanceof Map<?, ?>)) {

            return false;
        }


        Map<?, ?> firstMap =
                (Map<?, ?>) first;


        return Double.valueOf(0.0125)
                .equals(
                        firstMap.get(
                                "reuse_probability"
                        )
                );
    }


    private static boolean isEmptyArray(
            Object value) {

        return value instanceof List<?>
                && ((List<?>) value).isEmpty();
    }


    // =========================
    // JSON ERRORS
    // =========================

    private static void testJsonErrors() {

        section("JSON errors");


        check(
                "truncated object rejected",
                throwsJson("{\"a\":1")
        );


        check(
                "unterminated string rejected",
                throwsJson("{\"a\":\"oops}")
        );


        check(
                "trailing content rejected",
                throwsJson("{} extra")
        );


        check(
                "bad literal rejected",
                throwsJson("{\"a\":tru}")
        );


        check(
                "array is not an object",
                throwsJson("[1,2]")
        );
    }


    private static boolean throwsJson(
            String text) {

        try {

            Json.parseObject(text);

            return false;

        } catch (Json.JsonException e) {

            return true;
        }
    }


    // =========================
    // DEGRADATION
    // =========================

    private static void testClientDegradation() {

        section("degradation");


        // Port 1 should normally have nothing listening.
        MLClient dead =
                new MLClient(
                        "http://127.0.0.1:1",
                        100,
                        true
                );


        Map<String, double[]> candidates =
                sampleCandidates();


        check(
                "dead service returns no victims",
                dead.rankVictims(
                        candidates,
                        2
                ).isEmpty()
        );


        check(
                "failure recorded",
                dead.getFailureCount() == 1
        );


        check(
                "circuit opens after a failure",
                !dead.isCircuitClosed()
        );


        long before =
                System.currentTimeMillis();


        List<String> second =
                dead.rankVictims(
                        candidates,
                        2
                );


        long elapsed =
                System.currentTimeMillis()
                        - before;


        check(
                "open circuit returns immediately",
                second.isEmpty()
                        && elapsed < 50
        );


        check(
                "open circuit skips the call",
                dead.getFailureCount() == 1
        );


        check(
                "dead service reports unreachable",
                dead.health()
                        .startsWith("unreachable")
        );


        MLClient off =
                new MLClient(
                        "http://127.0.0.1:8000",
                        100,
                        false
                );


        check(
                "disabled client never calls",
                !off.isCircuitClosed()
                        && off.predict(candidates)
                                .isEmpty()
                        && off.getFailureCount() == 0
        );


        check(
                "disabled health says so",
                "disabled".equals(
                        off.health()
                )
        );


        MLClient live =
                new MLClient();


        check(
                "empty candidates need no call",
                live.rankVictims(
                        Map.<String, double[]>of(),
                        3
                ).isEmpty()
                        && live.getFailureCount() == 0
        );


        check(
                "zero count needs no call",
                live.rankVictims(
                        candidates,
                        0
                ).isEmpty()
                        && live.getFailureCount() == 0
        );


        check(
                "trailing slash trimmed",
                new MLClient(
                        "http://127.0.0.1:8000///",
                        100,
                        true
                )
                        .getBaseUrl()
                        .equals(
                                "http://127.0.0.1:8000"
                        )
        );
    }


    // =========================
    // REQUEST SHAPE
    // =========================

    /**
     * Checks that the generated request contains
     * properly named features.
     */
    private static void testRequestShape() {

        section("request shape");


        MLClient client =
                new MLClient(
                        "http://127.0.0.1:1",
                        100,
                        true
                );


        String body =
                buildRequestVia(
                        client,
                        sampleCandidates()
                );


        Map<String, Object> parsed =
                Json.parseObject(body);


        check(
                "count carried",
                Double.valueOf(2.0)
                        .equals(
                                parsed.get("count")
                        )
        );


        Object candidates =
                parsed.get("candidates");


        check(
                "two candidates sent",
                candidates instanceof List<?>
                        && ((List<?>) candidates)
                                .size() == 2
        );


        if (!(candidates instanceof List<?>)) {

            return;
        }


        List<?> candidateList =
                (List<?>) candidates;


        if (candidateList.isEmpty()
                || !(candidateList.get(0)
                        instanceof Map<?, ?>)) {

            return;
        }


        Map<?, ?> first =
                (Map<?, ?>) candidateList.get(0);


        check(
                "key carried",
                "hot:1".equals(
                        first.get("key")
                )
        );


        Object features =
                first.get("features");


        check(
                "all features named",
                features instanceof Map<?, ?>
                        && ((Map<?, ?>) features)
                                .size()
                                == MLFeatures.NAMES.length
        );


        if (!(features instanceof Map<?, ?>)) {

            return;
        }


        Map<?, ?> featureMap =
                (Map<?, ?>) features;


        boolean allPresent = true;


        for (String name :
                MLFeatures.NAMES) {

            if (!featureMap.containsKey(name)) {

                allPresent = false;

                break;
            }
        }


        check(
                "every feature name present",
                allPresent
        );


        // A wrong-length vector should be skipped.
        Map<String, double[]> mixed =
                new LinkedHashMap<>();


        mixed.put(
                "good",
                new double[
                        MLFeatures.NAMES.length
                ]
        );


        mixed.put(
                "short",
                new double[]{
                        1.0,
                        2.0
                }
        );


        mixed.put(
                "null",
                null
        );


        String filtered =
                buildRequestVia(
                        client,
                        mixed
                );


        Map<String, Object> filteredParsed =
                Json.parseObject(filtered);


        Object filteredCandidates =
                filteredParsed.get(
                        "candidates"
                );


        check(
                "malformed vectors dropped",
                filteredCandidates instanceof List<?>
                        && ((List<?>) filteredCandidates)
                                .size() == 1
        );
    }


    /**
     * Builds the request body through MLClient's
     * private method using reflection.
     */
    private static String buildRequestVia(
            MLClient client,
            Map<String, double[]> candidates) {

        try {

            java.lang.reflect.Method method =
                    MLClient.class.getDeclaredMethod(
                            "buildEvictRequest",
                            Map.class,
                            int.class
                    );


            method.setAccessible(true);


            return (String) method.invoke(
                    client,
                    candidates,
                    2
            );

        } catch (Exception e) {

            throw new RuntimeException(e);
        }
    }


    // =========================
    // LIVE SERVICE
    // =========================

    private static void testLiveService() {

        section("live service");


        MLClient client =
                new MLClient();


        String health =
                client.health();


        if (!health.startsWith("ready")) {

            System.out.println(
                    "  [SKIP] service not ready: "
                            + health
            );


            System.out.println(
                    "         start it with: "
                            + "python -m ml.service"
            );


            return;
        }


        System.out.println(
                "  service: " + health
        );


        Map<String, double[]> candidates =
                sampleCandidates();


        List<String> victims =
                client.rankVictims(
                        candidates,
                        1
                );


        check(
                "one victim returned",
                victims.size() == 1
        );


        if (victims.size() == 1) {

            check(
                    "victim is a candidate",
                    candidates.containsKey(
                            victims.get(0)
                    )
            );


            check(
                    "cold key is the victim",
                    "cold:7".equals(
                            victims.get(0)
                    )
            );
        }


        Map<String, Double> scores =
                client.predict(
                        candidates
                );


        check(
                "both keys scored",
                scores.size() == 2
        );


        boolean validScores = true;


        for (Double score :
                scores.values()) {

            if (score == null
                    || score < 0.0
                    || score > 1.0) {

                validScores = false;

                break;
            }
        }


        check(
                "scores are probabilities",
                validScores
        );


        if (scores.containsKey("hot:1")
                && scores.containsKey("cold:7")) {

            check(
                    "hot scores above cold",
                    scores.get("hot:1")
                            > scores.get("cold:7")
            );
        }


        check(
                "successes recorded",
                client.getSuccessCount() == 2
                        && client.getFailureCount() == 0
        );


        check(
                "circuit stayed closed",
                client.isCircuitClosed()
        );
    }


    // =========================
    // FIXTURES
    // =========================

    /**
     * Creates one hot key and one cold key using
     * KeyStats so the feature vectors are generated
     * exactly like they would be inside MiniRedis.
     */
    private static Map<String, double[]> sampleCandidates() {

        Map<String, double[]> candidates =
                new LinkedHashMap<>();


        long now =
                System.currentTimeMillis();


        // =========================
        // HOT KEY
        // =========================

        KeyStats hot =
                new KeyStats(
                        now - 60_000
                );


        for (int i = 0; i < 60; i++) {

            hot.record(
                    "GET",
                    now - 60_000
                            + i * 1000L,
                    40
            );
        }


        candidates.put(
                "hot:1",
                hot.toFeatureVector(now)
        );


        // =========================
        // COLD KEY
        // =========================

        KeyStats cold =
                new KeyStats(
                        now - 600_000
                );


        cold.record(
                "SET",
                now - 600_000,
                40
        );


        candidates.put(
                "cold:7",
                cold.toFeatureVector(now)
        );


        return candidates;
    }


    // =========================
    // TEST HARNESS
    // =========================

    private static void section(
            String name) {

        System.out.println(name);
    }


    private static void check(
            String name,
            boolean condition) {

        if (condition) {

            passed++;

            System.out.println(
                    "  [PASS] " + name
            );

        } else {

            failed++;

            System.out.println(
                    "  [FAIL] " + name
            );
        }
    }
}