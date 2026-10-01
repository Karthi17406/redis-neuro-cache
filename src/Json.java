import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small JSON reader and writer.
 *
 * MiniRedis is built with plain javac and no dependency
 * management, so pulling in Jackson or Gson is not an
 * option.
 *
 * Supported JSON types:
 *   object  -> Map<String, Object>
 *   array   -> List<Object>
 *   string  -> String
 *   number  -> Double
 *   boolean -> Boolean
 *   null    -> null
 */
public final class Json {

    // =========================
    // WRITING
    // =========================

    /**
     * Escapes a string and wraps it in quotes.
     */
    public static String quote(String value) {

        StringBuilder out = new StringBuilder();

        out.append('"');

        for (int i = 0; i < value.length(); i++) {

            char c = value.charAt(i);

            switch (c) {

                case '"':
                    out.append("\\\"");
                    break;

                case '\\':
                    out.append("\\\\");
                    break;

                case '\n':
                    out.append("\\n");
                    break;

                case '\r':
                    out.append("\\r");
                    break;

                case '\t':
                    out.append("\\t");
                    break;

                case '\b':
                    out.append("\\b");
                    break;

                case '\f':
                    out.append("\\f");
                    break;

                default:

                    if (c < 0x20) {
                        out.append(
                                String.format("\\u%04x", (int) c)
                        );
                    } else {
                        out.append(c);
                    }

                    break;
            }
        }

        out.append('"');

        return out.toString();
    }

    /**
     * Renders a double, keeping the output valid JSON.
     *
     * NaN and infinity have no JSON representation,
     * so they become null.
     */
    public static String number(double value) {

        if (Double.isNaN(value)
                || Double.isInfinite(value)) {

            return "null";
        }

        return String.valueOf(value);
    }


    // =========================
    // PARSING
    // =========================

    public static Object parse(String text) {

        Parser parser = new Parser(text);

        parser.skipWhitespace();

        Object value = parser.readValue();

        parser.skipWhitespace();

        if (!parser.atEnd()) {

            throw new JsonException(
                    "Trailing content at position "
                            + parser.position
            );
        }

        return value;
    }

    /**
     * Parses and casts to an object.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseObject(String text) {

        Object value = parse(text);

        if (!(value instanceof Map)) {

            throw new JsonException(
                    "Expected a JSON object but found "
                            + describe(value)
            );
        }

        return (Map<String, Object>) value;
    }

    private static String describe(Object value) {

        if (value == null) {
            return "null";
        }

        return value.getClass().getSimpleName();
    }

    public static class JsonException
            extends RuntimeException {

        public JsonException(String message) {
            super(message);
        }
    }


    // =========================
    // PARSER
    // =========================

    private static final class Parser {

        private final String text;

        private int position;

        Parser(String text) {
            this.text = text;
        }

        boolean atEnd() {
            return position >= text.length();
        }

        void skipWhitespace() {

            while (!atEnd()
                    && Character.isWhitespace(
                            text.charAt(position))) {

                position++;
            }
        }

        char peek() {

            if (atEnd()) {

                throw new JsonException(
                        "Unexpected end of JSON input"
                );
            }

            return text.charAt(position);
        }

        void expect(char expected) {

            if (atEnd()
                    || text.charAt(position) != expected) {

                throw new JsonException(
                        "Expected "
                                + expected
                                + " at position "
                                + position
                );
            }

            position++;
        }

        Object readValue() {

            skipWhitespace();

            char c = peek();

            switch (c) {

                case '{':
                    return readObject();

                case '[':
                    return readArray();

                case '"':
                    return readString();

                case 't':
                case 'f':
                    return readBoolean();

                case 'n':
                    return readNull();

                default:
                    return readNumber();
            }
        }

        Map<String, Object> readObject() {

            expect('{');

            Map<String, Object> result =
                    new LinkedHashMap<>();

            skipWhitespace();

            if (peek() == '}') {

                position++;

                return result;
            }

            while (true) {

                skipWhitespace();

                String key = readString();

                skipWhitespace();

                expect(':');

                result.put(key, readValue());

                skipWhitespace();

                char c = peek();

                if (c == ',') {

                    position++;

                    continue;
                }

                expect('}');

                return result;
            }
        }

        List<Object> readArray() {

            expect('[');

            List<Object> result =
                    new ArrayList<>();

            skipWhitespace();

            if (peek() == ']') {

                position++;

                return result;
            }

            while (true) {

                result.add(readValue());

                skipWhitespace();

                char c = peek();

                if (c == ',') {

                    position++;

                    continue;
                }

                expect(']');

                return result;
            }
        }

        String readString() {

            expect('"');

            StringBuilder out =
                    new StringBuilder();

            while (true) {

                if (atEnd()) {

                    throw new JsonException(
                            "Unterminated string"
                    );
                }

                char c =
                        text.charAt(position++);

                if (c == '"') {

                    return out.toString();
                }

                if (c != '\\') {

                    out.append(c);

                    continue;
                }

                if (atEnd()) {

                    throw new JsonException(
                            "Unterminated escape"
                    );
                }

                char escape =
                        text.charAt(position++);

                switch (escape) {

                    case '"':
                        out.append('"');
                        break;

                    case '\\':
                        out.append('\\');
                        break;

                    case '/':
                        out.append('/');
                        break;

                    case 'b':
                        out.append('\b');
                        break;

                    case 'f':
                        out.append('\f');
                        break;

                    case 'n':
                        out.append('\n');
                        break;

                    case 'r':
                        out.append('\r');
                        break;

                    case 't':
                        out.append('\t');
                        break;

                    case 'u': {

                        if (position + 4
                                > text.length()) {

                            throw new JsonException(
                                    "Truncated unicode escape"
                            );
                        }

                        String hex =
                                text.substring(
                                        position,
                                        position + 4
                                );

                        position += 4;

                        try {

                            out.append(
                                    (char) Integer.parseInt(
                                            hex,
                                            16
                                    )
                            );

                        } catch (NumberFormatException e) {

                            throw new JsonException(
                                    "Invalid unicode escape"
                            );
                        }

                        break;
                    }

                    default:

                        throw new JsonException(
                                "Unknown escape \\"
                                        + escape
                        );
                }
            }
        }

        Boolean readBoolean() {

            if (text.startsWith(
                    "true",
                    position)) {

                position += 4;

                return Boolean.TRUE;
            }

            if (text.startsWith(
                    "false",
                    position)) {

                position += 5;

                return Boolean.FALSE;
            }

            throw new JsonException(
                    "Invalid literal at position "
                            + position
            );
        }

        Object readNull() {

            if (text.startsWith(
                    "null",
                    position)) {

                position += 4;

                return null;
            }

            throw new JsonException(
                    "Invalid literal at position "
                            + position
            );
        }

        Double readNumber() {

            int start = position;

            if (!atEnd()
                    && text.charAt(position) == '-') {

                position++;
            }

            while (!atEnd()) {

                char c =
                        text.charAt(position);

                boolean partOfNumber =
                        (c >= '0' && c <= '9')
                                || c == '.'
                                || c == 'e'
                                || c == 'E'
                                || c == '+'
                                || c == '-';

                if (!partOfNumber) {
                    break;
                }

                position++;
            }

            if (start == position) {

                throw new JsonException(
                        "Invalid value at position "
                                + start
                );
            }

            try {

                return Double.valueOf(
                        text.substring(
                                start,
                                position
                        )
                );

            } catch (NumberFormatException e) {

                throw new JsonException(
                        "Invalid number at position "
                                + start
                );
            }
        }
    }


    private Json() {
        // Static helpers only
    }
}