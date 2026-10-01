public class Main {

    public static void main(String[] args) {

        // Create the database
        Database database =
                new Database();

        // Create the transaction manager
        TransactionManager transactionManager =
                new TransactionManager();

        // Create the command executor
        CommandExecutor executor =
                new CommandExecutor(
                        database,
                        transactionManager
                );

        // Give the executor to the command parser
        CommandParser parser =
                new CommandParser(executor);

        // Start the Mini Redis CLI
        parser.start();
    }
}