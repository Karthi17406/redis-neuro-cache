
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RedisServer {

    public static void main(String[] args) {

        int port = 6380;

        // Thread pool for handling multiple clients
        ExecutorService threadPool =
                Executors.newFixedThreadPool(20);

        // Shared Database
        Database database =
                new Database();

        // Transaction Manager
        TransactionManager transactionManager =
                new TransactionManager();

        // Command Executor
        CommandExecutor executor =
                new CommandExecutor(
                        database,
                        transactionManager
                );

        try {

            // Port 6380
            // Backlog = 200 pending connections
            ServerSocket serverSocket =
                    new ServerSocket(port, 200);

            System.out.println(
                    "MiniRedis+ Server started on port "
                            + port
            );

            int clientCounter = 1;

            while (true) {

                System.out.println(
                        "Waiting for client..."
                );

                Socket clientSocket =
                        serverSocket.accept();

                String clientId =
                        "Client-" + clientCounter++;

                System.out.println(
                        clientId + " connected!"
                );

                ClientHandler clientHandler =
                        new ClientHandler(
                                clientSocket,
                                executor,
                                clientId
                        );

                threadPool.execute(clientHandler);
            }

        } catch (IOException e) {

            e.printStackTrace();
        }
    }
}

