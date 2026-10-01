
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class ClientHandler implements Runnable {

    private Socket clientSocket;
    private CommandExecutor executor;
    private String clientId;

    public ClientHandler(
            Socket clientSocket,
            CommandExecutor executor,
            String clientId
    ) {
        this.clientSocket = clientSocket;
        this.executor = executor;
        this.clientId = clientId;
    }

    @Override
    public void run() {

        try {

            BufferedReader input = new BufferedReader(
                    new InputStreamReader(
                            clientSocket.getInputStream()
                    )
            );

            PrintWriter output = new PrintWriter(
                    clientSocket.getOutputStream(),
                    true
            );

            String command;

            while ((command = input.readLine()) != null) {

                command = command.trim();

                System.out.println(
                        "Received from " + clientId + ": " + command
                );

                if (command.isEmpty()) {
                    output.println("ERROR: Empty command");
                    continue;
                }

                if (command.equalsIgnoreCase("EXIT")) {

                    output.println("BYE");

                    System.out.println(
                            clientId + " disconnected."
                    );

                    break;
                }

                try {

                    String response =
                            executor.execute(command, clientId);

                    writeResponse(output, response);

                } catch (Exception e) {

                    output.println(
                            "ERROR: " + e.getMessage()
                    );
                }
            }

            clientSocket.close();
      } catch (IOException e) {

            System.out.println(
                    "Client handler error: " + e.getMessage()
            );
        }
    }

    /**
     * Writes the response to the client.
     *
     * Single-line responses are sent normally.
     * Multi-line responses are framed using:
     *
     * +TEXT n
     *
     * followed by exactly n lines.
     */
    private void writeResponse(
            PrintWriter output,
            String response) {

        String[] lines = response.split("\\R", -1);

        // Remove empty lines caused only by a trailing newline.
        int count = lines.length;

        while (count > 1 && lines[count - 1].isEmpty()) {
            count--;
        }

        // Single-line response
        if (count <= 1) {

            output.println(response.trim());

            return;
        }

        // Multi-line response
        output.println("+TEXT " + count);

        for (int i = 0; i < count; i++) {
            output.println(lines[i]);
        }
    }
}

