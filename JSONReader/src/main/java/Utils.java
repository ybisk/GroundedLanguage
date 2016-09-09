import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

public class Utils {
  public static final Pattern whitespace_pattern = Pattern.compile("\\s+");
  public static final Pattern dash_pattern = Pattern.compile("_");

  /**
   * Euclidean Distance (in block-lengths) if (x,y,z)
   * Quaternion Distance if (a,b,c,d) [norm of difference]
   */
  public static double distance(double[] A, double[] B) {
    double v = 0;
    for (int i = 0; i < A.length; ++i)
      v += Math.pow(A[i] - B[i], 2);
    v = Math.sqrt(v);
    if (A.length == 3) return v / 0.1524;
    else return v;
  }

  public static Double[][] readMatrix(List<String> matrix) {
    Double[][] array = new Double[matrix.size()][];
    for (int i = 0; i < matrix.size(); ++i) {
      array[i] = Arrays.asList(whitespace_pattern.split(matrix.get(i).trim())).stream().map(Double::parseDouble).toArray(Double[]::new);
    }
    return array;
  }

}
