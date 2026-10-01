import java.io.*;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ConcurrentTransactionTest {

    public static void main(String[] args) {

        String host = "localhost";
        int port = 6380;

        int numberOfClients = 10;

        ExecutorService pool =
                Executors.newFixedThreadPool(numberOfClients);

        for (int i = 1; i <= numberOfClients; i++) {

            int clientNumber = i;

            pool.execute(() -> {

                try {

                    Socket socket =
                            new Socket(host, port);

                    PrintWriter output =
                            new PrintWriter(
                                    socket.getOutputStream(),
                                    true
                            );

                    BufferedReader input =
                            new BufferedReader(
                                    new InputStreamReader(
                                            socket.getInputStream()
                                    )
                            );

                    String key =
                            "transactionClient" + clientNumber;

                    String value =
                            "value" + clientNumber;

                    System.out.println(
                            "Client-" + clientNumber +
                            " started transaction"
                    );

                    // BEGIN
                    output.println("BEGIN");

                    String beginResponse =
                            input.readLine();

                    System.out.println(
                            "Client-" + clientNumber +
                            " BEGIN -> " +
                            beginResponse
                    );

                    // SET
                    output.println(
                            "SET " + key + " " + value
                    );

                    String setResponse =
                            input.readLine();

                    System.out.println(
                            "Client-" + clientNumber +
                            " SET -> " +
                            setResponse
                    );

                    // GET inside transaction
                    output.println(
                            "GET " + key
                    );

                    String getResponse =
                            input.readLine();

                    System.out.println(
                            "Client-" + clientNumber +
                            " GET -> " +
                            getResponse
                    );

                    // Even clients COMMIT
                    // Odd clients ROLLBACK
                    if (clientNumber % 2 == 0) {

                        output.println("COMMIT");

                        String commitResponse =
                                input.readLine();

                        System.out.println(
                                "Client-" + clientNumber +
                                " COMMIT -> " +
                                commitResponse
                        );

                    } else {

                        output.println("ROLLBACK");

                        String rollbackResponse =
                                input.readLine();

                        System.out.println(
                                "Client-" + clientNumber +
                                " ROLLBACK -> " +
                                rollbackResponse
                        );
                    }

                    socket.close();

                } catch (IOException e) {

                    System.out.println(
                            "Client-" + clientNumber +
                            " FAILED: " +
                            e.getMessage()
                    );
                }
            });
        }

        pool.shutdown();

        System.out.println(
                "\nAll transaction clients submitted."
        );
    }
}
