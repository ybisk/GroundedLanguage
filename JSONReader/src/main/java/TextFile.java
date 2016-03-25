import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Stolen from https://github.com/ybisk/CCG-Induction
 */
public class TextFile {

  /**
   * Determines if a file is GZIPed or not based on filename and then creates
   * a buffered reader
   *
   * @param filename File to open for reading
   * @return A BufferedReader
   */
  public static BufferedReader Reader(String filename) {
    try {
      if (filename == null)
        throw new AssertionError("Null Filename");
      // Check if file exists in the classpath, otherwise load it locally
      InputStream in = new FileInputStream(new File(filename));
      if (filename.endsWith(".gz")) {
        return new BufferedReader(new InputStreamReader(new GZIPInputStream(in), "UTF-8"));
      }
      return new BufferedReader(new InputStreamReader(in, "UTF-8"));
    } catch (IOException exception) {
      exception.printStackTrace();
      throw new AssertionError("Invalid File " + filename);
    }
  }

  /**
   * Determines if a file is GZIPed or not based on filename and then creates
   * a buffered reader
   *
   * @param filename File to open for reading
   * @return A BufferedReader
   */
  public static List<String> Read(String filename) {
    try {
      ArrayList<String> file = new ArrayList<>();
      BufferedReader reader = TextFile.Reader(filename);
      String line;
      while ((line = reader.readLine()) != null)
        file.add(line);
      return file;
    } catch (IOException exception) {
      exception.printStackTrace();
      throw new AssertionError("Invalid File: "+ filename);
    }
  }

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
