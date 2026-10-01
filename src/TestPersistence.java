import java.util.HashMap;

public class TestPersistence {

    public static void main(String[] args) {

        // Create test data
        HashMap<String, String> database = new HashMap<>();
        HashMap<String, Long> expiryTime = new HashMap<>();

        database.put("name", "Karthi");
        database.put("language", "Java");

        // Create PersistenceManager
        PersistenceManager pm = new PersistenceManager();

        // Save data
        pm.saveData(database, expiryTime);

        System.out.println("Data saved!");

        // Load data
        HashMap<String, Object> loadedData = pm.loadData();

        HashMap<String, String> loadedDatabase =
                (HashMap<String, String>) loadedData.get("database");

        System.out.println("Loaded data:");
        System.out.println(loadedDatabase);
    }
}