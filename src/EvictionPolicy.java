/**
 * How the database chooses what to drop when it is full.
 *
 * ML is the point of the project, but it is deliberately
 * not the only option. LRU and LFU are the policies the
 * model has to beat, and keeping them selectable at
 * runtime means the comparison can be made on the same
 * server with the same workload rather than argued about
 * from offline numbers alone.
 */
public enum EvictionPolicy {

    /** Drop whatever was read least recently. */
    LRU,

    /** Drop whatever is read least often. */
    LFU,

    /** Ask the model, and fall back to LRU if it cannot answer. */
    ML,

    /** Never evict; let the database grow. */
    NONE;


    public static EvictionPolicy parse(String text) {

        if (text == null) {

            throw new IllegalArgumentException(
                    "No policy given");
        }

        return valueOf(text.trim().toUpperCase());
    }


    public static String available() {

        StringBuilder names = new StringBuilder();

        for (EvictionPolicy policy : values()) {

            if (names.length() > 0) {
                names.append(", ");
            }

            names.append(policy.name());
        }

        return names.toString();
    }
}
