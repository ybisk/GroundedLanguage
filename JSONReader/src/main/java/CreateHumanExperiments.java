import JSON.Note;
import JSON.Task;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by ybisk on 3/30/16.
 */
public class CreateHumanExperiments {
  static final GsonBuilder gsonBuilder = new GsonBuilder();

  public static Object convertWorld(double[][] current) {
    HashMap<String,Object>[] block_state = new HashMap[current.length];
    for (int i = 0; i < current.length; ++i) {
      block_state[i] = new HashMap<>();
      block_state[i].put("position", String.format("%f,%f,%f", current[i][0], current[i][1], current[i][2]));
      block_state[i].put("id", i+1);
    }
    return block_state;
  }

  public static String shape_params = "{\"shape_params\": {" +
      "\"face_4\": {\"color\": \"magenta\", \"orientation\": \"1\"}, " +
      "\"face_5\": {\"color\": \"yellow\", \"orientation\": \"1\"}, " +
      "\"face_6\": {\"color\": \"red\", \"orientation\": \"2\"}, " +
      "\"face_1\": {\"color\": \"blue\", \"orientation\": \"1\"}, " +
      "\"face_2\": {\"color\": \"green\", \"orientation\": \"1\"}, " +
      "\"face_3\": {\"color\": \"cyan\", \"orientation\": \"1\"}, " +
      "\"side_length\": 0.1524}";

  public static String[] blocks_params = new String[] {
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"adidas\", \"id\": 1}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"bmw\", \"id\": 2}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"burger king\", \"id\": 3}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"coca cola\", \"id\": 4}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"esso\", \"id\": 5}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"heineken\", \"id\": 6}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"hp\", \"id\": 7}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"mcdonalds\", \"id\": 8}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"mercedes benz\", \"id\": 9}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"nvidia\", \"id\": 10}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"pepsi\", \"id\": 11}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"shell\", \"id\": 12}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"sri\", \"id\": 13}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"starbucks\", \"id\": 14}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"stella artois\", \"id\": 15}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"target\", \"id\": 16}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"texaco\", \"id\": 17}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"toyota\", \"id\": 18}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"twitter\", \"id\": 19}",
      "{\"shape\": " + shape_params + ", \"type\": \"cube\", \"size\": 0.5}, \"name\": \"ups\", \"id\": 20}",

  };

  public static Object getMeta(int num, String decoration) {
    HashMap<String, Object> block_meta = new HashMap<>();
    block_meta.put("decoration", decoration);
    block_meta.put("blocks", new Map[num]);
    for (int i = 0; i < num; ++i) {
      Map map = LoadJSON.gsonReader.fromJson(blocks_params[i], Map.class);
      map.put("id", ((Double) map.get("id")).intValue());   // Force ID to be an integer
      ((Map[]) block_meta.get("blocks"))[i] = map;
    }
    return block_meta;
  }

  /**
   * Human Experiments require:
   * 1) World, Utterance
   * 2) Gold
   */
  public static strictfp void main(String[] args) throws Exception {
    Configuration.setConfiguration(args.length > 0 ? args[0] : "JSONReader/config.properties");
    ArrayList<Task> Test = LoadJSON.readJSON(Configuration.testing);

    Random random = new Random(20160330L);

    // Extract all world utterance pairs
    ArrayList<String> utterances = new ArrayList<>();
    ArrayList<double[][]> currentWorld = new ArrayList<>();
    ArrayList<double[][]> nextWorld = new ArrayList<>();
    ArrayList<String> decoration = new ArrayList<>();
    for (Task task : Test) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String utterance : note.notes) {
            utterances.add(utterance);
            currentWorld.add(task.states[note.start]);
            nextWorld.add(task.states[note.finish]);
            decoration.add(task.decoration);
          }
        }
      }
    }

    // Grab a random example
    int idx;
    double[][] current, next;
    String[] utterance;
    BufferedWriter BW;
    BufferedWriter goldBW;
    for (int i = 0; i < 25; ++i) {   // Number of examples
      idx = random.nextInt(utterances.size());
      utterance = CreateTrainingData.tokenize(utterances.remove(idx));
      current = currentWorld.remove(idx);
      next = nextWorld.remove(idx);
      BW = TextFile.Writer(String.format("%s_%d_world.json", decoration.get(idx), idx));
      goldBW = TextFile.Writer(String.format("%s_%d_gold.json", decoration.get(idx), idx));

      HashMap<String,Object> JSON = new HashMap<>();
      JSON.put("utterance", utterance);
      JSON.put("name", String.format("%s_%d_world.json", decoration.get(idx), idx));
      JSON.put("block_state", convertWorld(current));
      JSON.put("block_meta", getMeta(current.length, decoration.get(idx)));

      BW.write(gsonBuilder.disableHtmlEscaping().create().toJson(JSON) + "\n");
      BW.close();

      HashMap<String,Object> Gold = new HashMap<>();
      Gold.put("utternace", utterance);
      Gold.put("name", String.format("%s_%d_gold.json", decoration.get(idx), idx));
      Gold.put("current", current);
      Gold.put("next", next);
      goldBW.write(gsonBuilder.disableHtmlEscaping().create().toJson(Gold) + "\n");
      goldBW.close();
    }
  }
}
