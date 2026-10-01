import java.util.List;

public class CommandExecutor {

    private static final String NL =
            System.lineSeparator();

    private Database database;
    private TransactionManager transactionManager;

    // =========================
    // CONSTRUCTOR
    // =========================

    public CommandExecutor(
            Database database,
            TransactionManager transactionManager) {

        this.database = database;
        this.transactionManager = transactionManager;
    }

    // =========================
    // EXECUTE COMMAND
    // =========================

    public String execute(
            String commandLine,
            String clientId) {

        if (commandLine == null ||
                commandLine.trim().isEmpty()) {

            return "ERROR";
        }

        String[] parts =
                commandLine.trim().split("\\s+");

        String command =
                parts[0].toUpperCase();

        // =========================
        // SET
        // =========================

        if (command.equals("SET")) {

            if (parts.length < 3) {
                return "ERROR: SET key value";
            }

            String key = parts[1];

            StringBuilder valueBuilder =
                    new StringBuilder();

            for (int i = 2; i < parts.length; i++) {

                if (i > 2) {
                    valueBuilder.append(" ");
                }

                valueBuilder.append(parts[i]);
            }

            String value =
                    valueBuilder.toString();

            if (transactionManager.isInTransaction(clientId)) {

                transactionManager.addCommand(
                        clientId,
                        "SET",
                        new String[]{key, value}
                );

                return "OK";
            }

            database.set(key, value);

            return "OK";
        }

        // =========================
        // GET
        // =========================

        if (command.equals("GET")) {

            if (parts.length != 2) {
                return "ERROR: GET key";
            }

            String result =
                    database.get(parts[1]);

            if (result == null) {
                return "(nil)";
            }

            return result;
        }

        // =========================
        // DELETE
        // =========================

        if (command.equals("DELETE")) {

            if (parts.length != 2) {
                return "ERROR: DELETE key";
            }

            if (transactionManager.isInTransaction(clientId)) {

                transactionManager.addCommand(
                        clientId,
                        "DELETE",
                        new String[]{parts[1]}
                );

                return "OK";
            }

            boolean deleted =
                    database.delete(parts[1]);

            return deleted ? "1" : "0";
        }

        // =========================
        // EXISTS
        // =========================

        if (command.equals("EXISTS")) {

            if (parts.length != 2) {
                return "ERROR: EXISTS key";
            }

            return database.exists(parts[1])
                    ? "1"
                    : "0";
        }

        // =========================
        // KEYS
        // =========================

        if (command.equals("KEYS")) {

            if (parts.length != 1) {
                return "ERROR: KEYS";
            }

            return database.keys().toString();
        }

        // =========================
        // FLUSH
        // =========================

        if (command.equals("FLUSH")) {

            if (parts.length != 1) {
                return "ERROR: FLUSH";
            }

            if (transactionManager.isInTransaction(clientId)) {

                transactionManager.addCommand(
                        clientId,
                        "FLUSH",
                        new String[]{}
                );

                return "OK";
            }

            database.flush();

            return "OK";
        }

        // =========================
        // EXPIRE
        // =========================

        if (command.equals("EXPIRE")) {

            if (parts.length != 3) {
                return "ERROR: EXPIRE key seconds";
            }

            try {

                long seconds =
                        Long.parseLong(parts[2]);

                if (transactionManager
                        .isInTransaction(clientId)) {

                    transactionManager.addCommand(
                            clientId,
                            "EXPIRE",
                            new String[]{
                                    parts[1],
                                    parts[2]
                            }
                    );

                    return "1";
                }

                boolean expired =
                        database.expire(
                                parts[1],
                                seconds
                        );

                return expired ? "1" : "0";

            } catch (NumberFormatException e) {

                return "ERROR: seconds must be a number";
            }
        }

        // =========================
        // TTL
        // =========================

        if (command.equals("TTL")) {

            if (parts.length != 2) {
                return "ERROR: TTL key";
            }

            return String.valueOf(
                    database.ttl(parts[1])
            );
        }

        // =========================
        // BEGIN
        // =========================

        if (command.equals("BEGIN")) {

            if (parts.length != 1) {
                return "ERROR: BEGIN";
            }

            boolean began =
                    transactionManager.begin(clientId);

            return began
                    ? "OK"
                    : "ERROR: Transaction already active";
        }

        // =========================
        // COMMIT
        // =========================

        if (command.equals("COMMIT")) {

            if (parts.length != 1) {
                return "ERROR: COMMIT";
            }

            if (!transactionManager
                    .isInTransaction(clientId)) {

                return "ERROR: No active transaction";
            }

            List<String[]> pendingCommands =
                    transactionManager.commit(clientId);

            for (String[] pendingCommand :
                    pendingCommands) {

                String pendingName =
                        pendingCommand[0];

                String pendingArgs =
                        pendingCommand[1];

                String[] args;

                if (pendingArgs.isEmpty()) {
                    args = new String[]{};
                } else {
                    args =
                            pendingArgs.split(
                                    "\\s+"
                            );
                }

                StringBuilder rebuiltCommand =
                        new StringBuilder(
                                pendingName
                        );

                for (String arg : args) {

                    rebuiltCommand
                            .append(" ")
                            .append(arg);
                }

                execute(
                        rebuiltCommand.toString(),
                        clientId
                );
            }

            return "OK";
        }

        // =========================
        // ROLLBACK
        // =========================

        if (command.equals("ROLLBACK")) {

            if (parts.length != 1) {
                return "ERROR: ROLLBACK";
            }

            boolean rolledBack =
                    transactionManager.rollback(
                            clientId
                    );

            return rolledBack
                    ? "OK"
                    : "ERROR: No active transaction";
        }

        // =========================
        // MAXKEYS
        // =========================

        if (command.equals("MAXKEYS")) {

            if (parts.length == 1) {

                int limit = database.getMaxKeys();

                return limit == 0
                        ? "unlimited"
                        : String.valueOf(limit);
            }

            if (parts.length != 2) {
                return "ERROR: MAXKEYS [n]";
            }

            try {

                int limit = Integer.parseInt(parts[1]);

                if (limit < 0) {
                    return "ERROR: n must not be negative";
                }

                int evicted = database.setMaxKeys(limit);

                return evicted == 0
                        ? "OK"
                        : "OK, evicted " + evicted;

            } catch (NumberFormatException e) {

                return "ERROR: n must be a number";
            }
        }

        // =========================
        // POLICY
        // =========================

        if (command.equals("POLICY")) {

            EvictionManager manager =
                    database.getEvictionManager();

            if (parts.length == 1) {

                return manager.getPolicy().name();
            }

            if (parts.length != 2) {
                return "ERROR: POLICY ["
                        + EvictionPolicy.available()
                        + "]";
            }

            try {

                manager.setPolicy(
                        EvictionPolicy.parse(parts[1]));

                return "OK";

            } catch (IllegalArgumentException e) {

                return "ERROR: unknown policy, choices are "
                        + EvictionPolicy.available();
            }
        }

        // =========================
        // STATS
        // =========================

        if (command.equals("STATS")) {

            if (parts.length != 1) {
                return "ERROR: STATS";
            }

            StringBuilder stats = new StringBuilder();

            stats.append("keys: ")
                 .append(database.size());

            int limit = database.getMaxKeys();

            stats.append(NL + "maxkeys: ")
                 .append(limit == 0 ? "unlimited" : limit);

            stats.append(NL + "tracked keys: ")
                 .append(database.getStatsTracker().size());

            stats.append(NL)
                 .append(database.getEvictionManager()
                                 .describe());

            return stats.toString();
        }

        // =========================
        // HELP
        // =========================

        if (command.equals("HELP")) {

            if (parts.length != 1) {
                return "ERROR: HELP";
            }

                return "Available commands:\n"
                    + "SET key value\n"
                    + "GET key\n"
                    + "DELETE key\n"
                    + "EXISTS key\n"
                    + "KEYS\n"
                    + "FLUSH\n"
                    + "EXPIRE key seconds\n"
                    + "TTL key\n"
                    + "MAXKEYS [n]\n"
                    + "POLICY [LRU|LFU|ML|NONE]\n"
                    + "STATS\n"
                    + "BEGIN\n"
                    + "COMMIT\n"
                    + "ROLLBACK\n"
                    + "HELP\n"
                    + "EXIT";
        }

        // =========================
        // EXIT
        // =========================

        if (command.equals("EXIT")) {

            return "Goodbye!";
        }

        // =========================
        // UNKNOWN COMMAND
        // =========================

        return "ERROR: Unknown command";
    }
}