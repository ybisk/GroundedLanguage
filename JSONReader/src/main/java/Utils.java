import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

public class Utils {
  public static final Pattern whitespace_pattern = Pattern.compile("\\s+");
  public static final Pattern dash_pattern = Pattern.compile("_");

  /**
   * Euclidean Distance (in block-lengths)
   */
  public static double distance(double[] A, double[] B) {
    return Math.sqrt(Math.pow(A[0] - B[0], 2) + Math.pow(A[1] - B[1], 2) + Math.pow(A[2] - B[2], 2)) / 0.1524;
  }

  public static Double[][] readMatrix(List<String> matrix) {
    Double[][] array = new Double[matrix.size()][];
    for (int i = 0; i < matrix.size(); ++i) {
      array[i] = Arrays.asList(whitespace_pattern.split(matrix.get(i).trim())).stream().map(Double::parseDouble).toArray(Double[]::new);
    }
    return array;
  }

}
