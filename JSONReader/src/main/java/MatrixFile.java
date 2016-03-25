import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class MatrixFile {

  /**
   * @param filename File to open for writing
   * @return A new buffered Writer
   */
  public static BufferedWriter Writer(String filename) {
    try {
      if (filename.endsWith(".gz")) {
        return new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(
            new FileOutputStream(new File(filename))), "UTF-8"));
      }
      return new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
          new File(filename)), "UTF-8"));
    } catch (IOException exception) {
      exception.printStackTrace();
      throw new AssertionError("Invalid File");
    }
  }

}
