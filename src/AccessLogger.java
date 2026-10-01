import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Writes one row per key access to a CSV file.
 *
 * The rows are the raw material for the ML pipeline:
 * feature building, label creation and model training
 * all start from this file.
 *
 * Row format (no header, to stay compatible with the
 * logs that were already collected):
 *
 *     key,operation,timestamp,hit,valueSize
 *
 * Older rows only have the first three columns.
 * The Python loader pads them, so both formats mix freely.
 */
public class AccessLogger {

    // =========================
    // CONFIGURATION
    // =========================

    private static final String DEFAULT_FILE =
            "access_log.csv";

    // Rows buffered before the data reaches the disk
    private static final int FLUSH_EVERY = 64;

    // One writer is shared by every AccessLogger, so
    // several Database objects can log to one file
    // without corrupting each other's rows.
    private static BufferedWriter writer;

    private static String openFile;

    private static int rowsSinceFlush;

    private static boolean disabled;

    private static boolean shutdownHookInstalled;


    // =========================
    // FILE NAME
    // =========================

    /**
     * The log file can be redirected with
     * -Dminiredis.accesslog=some/other/file.csv
     * which keeps tests from polluting the real log.
     */
    private static String fileName() {

        return System.getProperty(
                "miniredis.accesslog",
                DEFAULT_FILE
        );
    }


    // =========================
    // LOG
    // =========================

    /**
     * Logs an access without hit/miss information.
     */
    public void log(
            String key,
            String operation) {

        log(key, operation, null, -1);
    }


    /**
     * Logs an access.
     *
     * @param hit       whether the key was present,
     *                  or null when it does not apply
     * @param valueSize size of the stored value,
     *                  or -1 when it is unknown
     */
    public void log(
            String key,
            String operation,
            Boolean hit,
            int valueSize) {

        synchronized (AccessLogger.class) {

            if (disabled) {
                return;
            }

            try {

                openWriter();

                writer.write(escape(key));
                writer.write(',');
                writer.write(escape(operation));
                writer.write(',');
                writer.write(String.valueOf(
                        System.currentTimeMillis()));
                writer.write(',');
                writer.write(
                        hit == null
                                ? ""
                                : (hit ? "1" : "0"));
                writer.write(',');
                writer.write(String.valueOf(valueSize));
                writer.write('\n');

                rowsSinceFlush++;

                if (rowsSinceFlush >= FLUSH_EVERY) {

                    writer.flush();
                    rowsSinceFlush = 0;
                }

            } catch (IOException e) {

                System.out.println(
                        "Error writing access log: "
                        + e.getMessage()
                );

                // A broken log must never take the
                // database down with it.
                closeQuietly();
                disabled = true;
            }
        }
    }


    // =========================
    // OPEN WRITER
    // =========================

    private static void openWriter() throws IOException {

        String target = fileName();

        // Reopen when the target file changed
        if (writer != null
                && !target.equals(openFile)) {

            closeQuietly();
        }

        if (writer != null) {
            return;
        }

        File file = new File(target);

        File parent = file.getParentFile();

        if (parent != null) {
            parent.mkdirs();
        }

        writer = new BufferedWriter(
                new FileWriter(file, true)
        );

        openFile = target;
        rowsSinceFlush = 0;

        installShutdownHook();
    }


    // =========================
    // SHUTDOWN HOOK
    // =========================

    /**
     * Buffered rows would be lost when the JVM stops,
     * so the last partial batch is flushed on exit.
     */
    private static void installShutdownHook() {

        if (shutdownHookInstalled) {
            return;
        }

        shutdownHookInstalled = true;

        Runtime.getRuntime().addShutdownHook(
                new Thread(AccessLogger::flush)
        );
    }


    // =========================
    // FLUSH
    // =========================

    /**
     * Pushes buffered rows to disk. Call this before
     * reading the log from another process.
     */
    public static void flush() {

        synchronized (AccessLogger.class) {

            if (writer == null) {
                return;
            }

            try {

                writer.flush();
                rowsSinceFlush = 0;

            } catch (IOException e) {

                System.out.println(
                        "Error flushing access log: "
                        + e.getMessage()
                );
            }
        }
    }


    // =========================
    // CLOSE
    // =========================

    /**
     * Flushes and closes the log, and lets a later
     * write reopen it. Mainly used by the tests.
     */
    public static void close() {

        synchronized (AccessLogger.class) {

            flush();
            closeQuietly();

            disabled = false;
        }
    }


    private static void closeQuietly() {

        if (writer != null) {

            try {
                writer.close();
            } catch (IOException ignored) {
                // Nothing useful can be done here
            }

            writer = null;
            openFile = null;
        }
    }


    // =========================
    // CSV ESCAPING
    // =========================

    /**
     * Keys are user supplied, so a key containing a
     * comma, quote or newline must not break the CSV.
     */
    private static String escape(String field) {

        if (field == null) {
            return "";
        }

        boolean needsQuotes =
                field.indexOf(',') >= 0
                || field.indexOf('"') >= 0
                || field.indexOf('\n') >= 0
                || field.indexOf('\r') >= 0;

        if (!needsQuotes) {
            return field;
        }

        return "\""
                + field.replace("\"", "\"\"")
                + "\"";
    }
}
