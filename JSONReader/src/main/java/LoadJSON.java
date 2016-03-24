import JSON.Task;
import com.google.gson.Gson;

import java.io.*;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;

public class LoadJSON {
  private static final Gson gsonReader = new Gson();

  public static ArrayList<Task> readJSON(String filename) throws IOException {
    BufferedReader BR = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(new File(filename))), "UTF-8"));
    ArrayList<Task> Tasks = new ArrayList<>();
    String line;
    while ((line = BR.readLine()) != null)
      Tasks.add(gsonReader.fromJson(line, Task.class));
    return Tasks;
  }

  public static strictfp void main(String[] args) throws Exception {
    ArrayList<Task> Dev = readJSON(args[0]);
  }
}
