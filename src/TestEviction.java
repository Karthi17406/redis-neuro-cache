import java.io.File;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tests capacity bounded eviction.
 *
 * Three properties matter, in this order:
 *
 *   1. A database with no bound behaves exactly as it
 *      did before eviction existed. This is the
 *      regression that would be worst to introduce.
 *   2. A bound is never exceeded, whatever the policy
 *      decides or fails to decide.
 *   3. Each policy drops the key it claims to drop.
 *
 * The ML policy is tested twice, once against a dead
 * service and once against a live one, because falling
 * back correctly is as much a requirement as predicting
 * correctly.
 */
public class TestEviction {

    private static int passed;

    private static int failed;


    public static void main(String[] args) throws Exception {

        System.out.println("Testing eviction\n");

        testUnboundedIsUnchanged();
        testBoundIsHonoured();
        testLruChoosesStalest();
        testLfuChoosesRarest();
        testNoneNeverEvicts();
        testMlFallsBackWhenServiceIsDown();
        testMlUsesTheModelWhenAvailable();
        testEvictionForgetsStatistics();
        testSetMaxKeysShrinks();

        System.out.println(
                "\n" + passed + " passed, " + failed + " failed");

        if (failed > 0) {
            System.exit(1);
        }
    }


    // =========================
    // NO BOUND
    // =========================

    private static void testUnboundedIsUnchanged() {

        section("unbounded");

        Database db = freshDatabase("unbounded");

        check("default is unlimited", db.getMaxKeys() == 0);

        for (int i = 0; i < 50; i++) {

            db.set("key:" + i, "value" + i);
        }

        check("nothing evicted", db.size() == 50);

        check("every key still readable",
                "value0".equals(db.get("key:0"))
                        && "value49".equals(db.get("key:49")));

        check("eviction counter untouched",
                db.getEvictionManager()
                        .getEvictionCount() == 0);
    }


    // =========================
    // BOUND
    // =========================

    private static void testBoundIsHonoured() {

        section("bound");

        Database db = freshDatabase("bound");

        db.setMaxKeys(10);

        for (int i = 0; i < 40; i++) {

            db.set("key:" + i, "value" + i);
        }

        check("size held at the bound", db.size() == 10);

        check("evictions counted",
                db.getEvictionManager()
                        .getEvictionCount() == 30);

        // Overwriting an existing key adds nothing, so
        // it must not trigger an eviction.
        long before =
                db.getEvictionManager().getEvictionCount();

        String survivor = db.keys().iterator().next();

        db.set(survivor, "replacement");

        check("overwrite evicts nothing",
                db.getEvictionManager()
                        .getEvictionCount() == before
                        && db.size() == 10);
    }


    // =========================
    // LRU
    // =========================

    private static void testLruChoosesStalest() throws Exception {

        section("LRU");

        Database db = freshDatabase("lru");

        db.getEvictionManager()
                .setPolicy(EvictionPolicy.LRU);

        db.setMaxKeys(3);

        db.set("a", "1");
        Thread.sleep(20);

        db.set("b", "2");
        Thread.sleep(20);

        db.set("c", "3");
        Thread.sleep(20);

        // Touching a and b leaves c as the stalest.
        db.get("a");
        Thread.sleep(20);

        db.get("b");
        Thread.sleep(20);

        db.set("d", "4");

        check("stalest key evicted", !db.exists("c"));

        check("recently read keys kept",
                db.exists("a")
                        && db.exists("b")
                        && db.exists("d"));
    }


    // =========================
    // LFU
    // =========================

    private static void testLfuChoosesRarest() throws Exception {

        section("LFU");

        Database db = freshDatabase("lfu");

        db.getEvictionManager()
                .setPolicy(EvictionPolicy.LFU);

        db.setMaxKeys(3);

        db.set("hot", "1");
        db.set("warm", "2");
        db.set("rare", "3");

        for (int i = 0; i < 30; i++) {
            db.get("hot");
        }

        for (int i = 0; i < 10; i++) {
            db.get("warm");
        }

        // rare was written once and never read since.
        Thread.sleep(20);

        db.set("new", "4");

        check("rarest key evicted", !db.exists("rare"));

        check("frequent keys kept",
                db.exists("hot") && db.exists("warm"));
    }


    // =========================
    // NONE
    // =========================

    private static void testNoneNeverEvicts() {

        section("NONE");

        Database db = freshDatabase("none");

        db.getEvictionManager()
                .setPolicy(EvictionPolicy.NONE);

        db.setMaxKeys(5);

        for (int i = 0; i < 20; i++) {

            db.set("key:" + i, "value" + i);
        }

        // NONE is an explicit instruction not to drop
        // anything, so the bound is deliberately allowed
        // to be exceeded rather than quietly enforced.
        check("bound not enforced", db.size() == 20);

        check("nothing evicted",
                db.getEvictionManager()
                        .getEvictionCount() == 0);
    }


    // =========================
    // ML WITHOUT A SERVICE
    // =========================

    /**
     * The important one. A model service that is not
     * running must cost the cache nothing beyond one
     * timeout, and must not stop it honouring its bound.
     */
    private static void testMlFallsBackWhenServiceIsDown()
            throws Exception {

        section("ML, service down");

        Database db = freshDatabase("mldown");

        db.setEvictionManager(
                new EvictionManager(
                        EvictionPolicy.ML,
                        new MLClient(
                                "http://127.0.0.1:1",
                                100,
                                true)));

        db.setMaxKeys(3);

        db.set("a", "1");
        Thread.sleep(20);

        db.set("b", "2");
        Thread.sleep(20);

        db.set("c", "3");
        Thread.sleep(20);

        db.get("b");
        Thread.sleep(20);

        db.get("c");
        Thread.sleep(20);

        long before = System.currentTimeMillis();

        db.set("d", "4");

        long elapsed = System.currentTimeMillis() - before;

        check("bound still honoured", db.size() == 3);

        check("fell back to LRU", !db.exists("a"));

        check("counted as a fallback",
                db.getEvictionManager()
                        .getFallbackDecisions() == 1
                        && db.getEvictionManager()
                                .getMlDecisions() == 0);

        // The first write pays one timeout, after which
        // the circuit is open.
        long open = System.currentTimeMillis();

        db.set("e", "5");

        long openElapsed = System.currentTimeMillis() - open;

        check("first failure is bounded", elapsed < 2000);

        check("later writes do not wait at all",
                openElapsed < 50);

        check("still bounded", db.size() == 3);
    }


    // =========================
    // ML WITH A SERVICE
    // =========================

    private static void testMlUsesTheModelWhenAvailable()
            throws Exception {

        section("ML, service up");

        MLClient client = new MLClient();

        String health = client.health();

        if (!health.startsWith("ready")) {

            System.out.println(
                    "  [SKIP] service not ready: " + health);

            return;
        }

        System.out.println("  service: " + health);

        Database db = freshDatabase("mlup");

        db.setEvictionManager(
                new EvictionManager(
                        EvictionPolicy.ML, client));

        db.setMaxKeys(3);

        db.set("hot", "1");
        db.set("warm", "2");
        db.set("oneshot", "3");

        // A clear reuse pattern, so a working model and
        // a working fallback would agree. What is being
        // tested here is that the ML path runs and is
        // counted as such.
        for (int i = 0; i < 40; i++) {

            db.get("hot");

            if (i % 4 == 0) {
                db.get("warm");
            }
        }

        Thread.sleep(20);

        db.set("new", "4");

        check("bound honoured", db.size() == 3);

        check("model was consulted",
                db.getEvictionManager().getMlDecisions() == 1);

        check("no fallback needed",
                db.getEvictionManager()
                        .getFallbackDecisions() == 0);

        check("hot key survived", db.exists("hot"));

        check("service calls succeeded",
                client.getSuccessCount() > 0);
    }


    // =========================
    // STATISTICS
    // =========================

    private static void testEvictionForgetsStatistics() {

        section("statistics");

        Database db = freshDatabase("stats");

        db.setMaxKeys(5);

        for (int i = 0; i < 20; i++) {

            db.set("key:" + i, "value" + i);
        }

        check("tracker matches the database",
                db.getStatsTracker().size() == db.size());

        boolean onlyResident = true;

        for (String tracked
                : db.getStatsTracker().trackedKeys()) {

            if (!db.exists(tracked)) {
                onlyResident = false;
            }
        }

        check("no statistics for evicted keys", onlyResident);

        check("evicted keys read as missing",
                db.get("key:0") == null);
    }


    // =========================
    // SHRINKING
    // =========================

    private static void testSetMaxKeysShrinks() {

        section("shrinking");

        Database db = freshDatabase("shrink");

        for (int i = 0; i < 30; i++) {

            db.set("key:" + i, "value" + i);
        }

        check("all present before the bound",
                db.size() == 30);

        int evicted = db.setMaxKeys(10);

        check("shrink reports what it dropped",
                evicted == 20);

        check("size now at the bound", db.size() == 10);

        // Going back to unlimited must not resurrect
        // anything, but must stop dropping.
        db.setMaxKeys(0);

        db.set("fresh", "value");

        check("unlimited stops evicting", db.size() == 11);
    }


    // =========================
    // HARNESS
    // =========================

    /**
     * A database with its own data file and access log,
     * so the tests cannot see each other and cannot
     * disturb a real deployment.
     */
    private static Database freshDatabase(String name) {

        File directory =
                new File(
                        System.getProperty(
                                "java.io.tmpdir"),
                        "miniredis-eviction");

        directory.mkdirs();

        File data = new File(directory, name + ".db");
        File log = new File(directory, name + ".csv");

        data.delete();
        log.delete();

        System.setProperty(
                "miniredis.datafile", data.getPath());

        System.setProperty(
                "miniredis.accesslog", log.getPath());

        return new Database();
    }


    private static void section(String name) {

        System.out.println(name);
    }


    private static void check(String name, boolean condition) {

        if (condition) {

            passed++;
            System.out.println("  [PASS] " + name);

        } else {

            failed++;
            System.out.println("  [FAIL] " + name);
        }
    }
}
