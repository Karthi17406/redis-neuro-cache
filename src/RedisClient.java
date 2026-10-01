
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class RedisClient {

    public static void main(String[] args) {

        String host = "localhost";
        int port = 6380;

        try {

            // Connect to MiniRedis+ Server
            Socket socket = new Socket(host, port);

            System.out.println("=================================");
            System.out.println("        MiniRedis+ Client");
            System.out.println("=================================");
            System.out.println("Connected to MiniRedis+ Server!");
            System.out.println("Server: " + host + ":" + port);
            System.out.println();

            System.out.println("Available commands:");
            System.out.println("SET key value");
            System.out.println("GET key");
            System.out.println("DELETE key");
            System.out.println("EXISTS key");
            System.out.println("KEYS");
            System.out.println("FLUSH");
            System.out.println("EXPIRE key seconds");
            System.out.println("TTL key");
            System.out.println("BEGIN");
            System.out.println("COMMIT");
            System.out.println("ROLLBACK");
            System.out.println("HELP");
            System.out.println("EXIT");
            System.out.println();

            // Send commands to server
            PrintWriter output = new PrintWriter(
                    socket.getOutputStream(),
                    true
            );

            // Receive responses from server
            BufferedReader input = new BufferedReader(
                    new InputStreamReader(
                            socket.getInputStream()
                    )
            );

            // Read commands from keyboard
            BufferedReader console = new BufferedReader(
                    new InputStreamReader(System.in)
            );

            while (true) {

                System.out.print("MiniRedis+> ");

                String command = console.readLine();

                // If input is closed
                if (command == null) {
                    break;
                }

                // Remove unnecessary spaces
                command = command.trim();

                // Ignore empty commands
                if (command.isEmpty()) {
                    continue;
                }

                // Send command to server
                output.println(command);

                // EXIT command
                if (command.equalsIgnoreCase("EXIT")) {

                    String response = input.readLine();

                    if (response != null) {
                        System.out.println("Server: " + response);
                    }

                    break;
                }

                // Receive response from server
                String response = input.readLine();

                if (response == null) {

                    System.out.println(
                            "Server disconnected."
                    );

                    break;
                }

                System.out.println("Server: " + response);
            }

            // Close everything
            console.close();
            input.close();
            output.close();
            socket.close();

            System.out.println();
            System.out.println(
                    "Disconnected from MiniRedis+ Server."
            );

        } catch (IOException e) {

            System.out.println();
            System.out.println(
                    "Unable to connect to MiniRedis+ Server."
            );

            System.out.println(
                    "Make sure RedisServer is running on port "
                    + port + "."
            );

            System.out.println();

            e.printStackTrace();
        }
    }
}

