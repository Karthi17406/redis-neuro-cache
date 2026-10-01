import java.io.*;
import java.util.HashMap;
import java.util.Iterator;

public class PersistenceManager {

    // File where data is stored
    private static final String DEFAULT_DATA_FILE =
            "mini_redis.db";

    /**
     * The data file can be redirected with
     * -Dminiredis.datafile=some/other/file.db so that
     * tests and workload generation never overwrite a
     * real database.
     */
    private static String dataFile() {

        return System.getProperty(
                "miniredis.datafile",
                DEFAULT_DATA_FILE
        );
    }

    // =========================
    // SAVE DATA
    // =========================

    public synchronized void saveData(
            HashMap<String, String> database,
            HashMap<String, Long> expiryTime) {

        try (
                FileOutputStream fileOutput =
                        new FileOutputStream(dataFile());

                ObjectOutputStream objectOutput =
                        new ObjectOutputStream(fileOutput)
        ) {

            // Save database
            objectOutput.writeObject(database);

            // Save expiration information
            objectOutput.writeObject(expiryTime);

        } catch (IOException e) {

            System.out.println(
                    "Error saving data: " + e.getMessage()
            );
        }
    }

    // =========================
    // LOAD DATA
    // =========================

    @SuppressWarnings("unchecked")
    public synchronized HashMap<String, Object> loadData() {

        HashMap<String, Object> result =
                new HashMap<>();

        File file =
                new File(dataFile());

        // File does not exist on first run
        if (!file.exists()) {
            return result;
        }

        try (
                FileInputStream fileInput =
                        new FileInputStream(dataFile());

                ObjectInputStream objectInput =
                        new ObjectInputStream(fileInput)
        ) {

            HashMap<String, String> database =
                    (HashMap<String, String>)
                            objectInput.readObject();

            HashMap<String, Long> expiryTime =
                    (HashMap<String, Long>)
                            objectInput.readObject();

            // Remove expired keys
            long currentTime =
                    System.currentTimeMillis();

            Iterator<String> iterator =
                    expiryTime.keySet().iterator();

            while (iterator.hasNext()) {

                String key =
                        iterator.next();

                long expiry =
                        expiryTime.get(key);

                if (currentTime >= expiry) {

                    database.remove(key);
                    iterator.remove();
                }
            }

            result.put("database", database);
            result.put("expiryTime", expiryTime);

        } catch (IOException | ClassNotFoundException e) {

            System.out.println(
                    "Error loading data: " + e.getMessage()
            );
        }

        return result;
    }
}