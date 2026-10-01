public class TestDatabase {

    public static void main(String[] args) {

        Database db = new Database();

        // =========================
        // SET
        // =========================

        db.set("name", "Karthi");
        db.set("language", "Java");


        // =========================
        // GET
        // =========================

        System.out.println(
                "GET name: " + db.get("name")
        );


        // =========================
        // EXISTS
        // =========================

        System.out.println(
                "EXISTS name: " + db.exists("name")
        );


        // =========================
        // KEYS
        // =========================

        System.out.println(
                "KEYS: " + db.keys()
        );


        // =========================
        // EXPIRE
        // =========================

        System.out.println(
                "EXPIRE name: " + db.expire("name", 5)
        );


        // =========================
        // TTL
        // =========================

        System.out.println(
                "TTL name: " + db.ttl("name")
        );


        // =========================
        // DELETE
        // =========================

        System.out.println(
                "DELETE language: " + db.delete("language")
        );


        // =========================
        // KEYS AFTER DELETE
        // =========================

        System.out.println(
                "KEYS after delete: " + db.keys()
        );


        // =========================
        // PERSISTENCE TEST
        // =========================

        System.out.println(
                "\n--- Persistence Test ---"
        );

        // Create a NEW Database object.
        // This simulates restarting Mini Redis.
        Database db2 = new Database();

        // Check whether the saved data was loaded.
        System.out.println(
                "Data after creating new Database: "
                + db2.get("name")
        );
    }
}