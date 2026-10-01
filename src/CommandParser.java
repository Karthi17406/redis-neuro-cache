import java.util.Scanner;

public class CommandParser {

    private CommandExecutor executor;

    // Constructor
    public CommandParser(CommandExecutor executor) {
        this.executor = executor;
    }

    // Start the CLI
    public void start() {

        Scanner scanner = new Scanner(System.in);

        System.out.println("================================");
        System.out.println("       Mini Redis Started");
        System.out.println("================================");
        System.out.println("Type HELP to see available commands.");
        System.out.println("Type EXIT to quit.");
        System.out.println();

        while (true) {

            System.out.print("MiniRedis> ");

            String input = scanner.nextLine().trim();

            // Ignore empty input
            if (input.isEmpty()) {
                continue;
            }

            // EXIT is handled by the parser
            if (input.equalsIgnoreCase("EXIT")) {

                System.out.println("Goodbye!");
                break;
            }

            // Send command to CommandExecutor
            String response = executor.execute(input, "CLI");

            // Print response
            System.out.println(response);
        }

        scanner.close();
    }
}