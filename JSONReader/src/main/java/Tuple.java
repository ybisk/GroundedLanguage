public class Tuple<T> implements Comparable<Object> {
  private final T contents;
  private final double value;

  /**
   * Create an immutable object which sorts by the double's value
   * @param object Non-sortable object
   * @param value Value to sort by
   */
  public Tuple(T object, double value) {
    contents = object;
    this.value = value;
  }

  /**
   * Returns the sorted content
   * @return  object
   */
  public T content() {
    return contents;
  }

  /**
   * Value of the object
   * @return value
   */
  public double value() {
    return value;
  }

  @SuppressWarnings("unchecked")
  public int compareTo(Object arg0) {
    return (int) Math.signum(((Tuple<T>) arg0).value - value);
  }
}

