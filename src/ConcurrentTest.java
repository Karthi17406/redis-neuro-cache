import java.io.*;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ConcurrentTest {

    public static void main(String[] args) {

        String host = "localhost";
        int port = 6380;
        int numberOfClients = 100;

        ExecutorService pool =
                Executors.newFixedThreadPool(100);

        for (int i = 1; i <= numberOfClients; i++) {

            int clientNumber = i;

            pool.execute(() -> {

                try {

                    Socket socket = new Socket(host, port);

                    PrintWriter output =
                            new PrintWriter(
                                    socket.getOutputStream(), true);

                    BufferedReader input =
                            new BufferedReader(
                                    new InputStreamReader(
                                            socket.getInputStream()));

                    String key = "client" + clientNumber;
                    String value = "value" + clientNumber;

                    output.println("SET " + key + " " + value);
                    String setResponse = input.readLine();

                    System.out.println(
                            "Client-" + clientNumber +
                            " SET -> " + setResponse);

                    output.println("GET " + key);
                    String getResponse = input.readLine();

                    System.out.println(
                            "Client-" + clientNumber +
                            " GET -> " + getResponse);

                    socket.close();

                } catch (Exception e) {

                    System.out.println(
                            "Client-" + clientNumber +
                            " FAILED: " + e.getMessage());
                }
            });
        }

        pool.shutdown();
    }
}