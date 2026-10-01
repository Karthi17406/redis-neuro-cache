public class AccessLogGenerator {

    public static void main(String[] args) {

        Database database = new Database();

        // Create some keys
        database.set("user1", "Karthika");
        database.set("user2", "Alice");
        database.set("user3", "Bob");
        database.set("product1", "Laptop");
        database.set("product2", "Phone");

        // Simulate repeated access
        for (int i = 0; i < 100; i++) {

            // user1 is accessed very frequently
            database.get("user1");

            // user2 is accessed sometimes
            if (i % 3 == 0) {
                database.get("user2");
            }

            // user3 is accessed less frequently
            if (i % 10 == 0) {
                database.get("user3");
            }

            // product1 is accessed frequently
            if (i % 2 == 0) {
                database.get("product1");
            }

            // product2 is rarely accessed
            if (i % 20 == 0) {
                database.get("product2");
            }

            // Small delay so timestamps are different
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        System.out.println(
                "Access log generation completed!"
        );
    }
}
