import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TransactionManager {

    // Each client has its own transaction
    private Map<String, Boolean> clientTransactions =
            new HashMap<>();

    // Each client has its own pending commands
    private Map<String, List<String[]>> clientCommands =
            new HashMap<>();

    // =========================
    // BEGIN
    // =========================

    public synchronized boolean begin(String clientId) {

        if (clientTransactions.getOrDefault(clientId, false)) {
            return false;
        }

        clientTransactions.put(clientId, true);

        clientCommands.put(
                clientId,
                new ArrayList<>()
        );

        return true;
    }

    // =========================
    // ADD COMMAND
    // =========================

    public synchronized void addCommand(
            String clientId,
            String command,
            String[] args) {

        if (clientTransactions.getOrDefault(clientId, false)) {

            clientCommands
                    .get(clientId)
                    .add(new String[]{
                            command,
                            String.join(" ", args)
                    });
        }
    }

    // =========================
    // CHECK TRANSACTION
    // =========================

    public synchronized boolean isInTransaction(
            String clientId) {

        return clientTransactions.getOrDefault(
                clientId,
                false
        );
    }

    // =========================
    // GET COMMANDS
    // =========================

    public synchronized List<String[]> getCommands(
            String clientId) {

        return new ArrayList<>(
                clientCommands.getOrDefault(
                        clientId,
                        new ArrayList<>()
                )
        );
    }

    // =========================
    // COMMIT
    // =========================

    public synchronized List<String[]> commit(
            String clientId) {

        if (!clientTransactions.getOrDefault(
                clientId,
                false)) {

            return new ArrayList<>();
        }

        List<String[]> pendingCommands =
                new ArrayList<>(
                        clientCommands.get(clientId)
                );

        clientCommands.remove(clientId);
        clientTransactions.remove(clientId);

        return pendingCommands;
    }

    // =========================
    // ROLLBACK
    // =========================

    public synchronized boolean rollback(
            String clientId) {

        if (!clientTransactions.getOrDefault(
                clientId,
                false)) {

            return false;
        }

        clientCommands.remove(clientId);
        clientTransactions.remove(clientId);

        return true;
    }
}