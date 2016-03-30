import JSON.Note;
import JSON.Task;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.StringUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class CreateTrainingData {
  static MaxentTagger tagger = new MaxentTagger(MaxentTagger.DEFAULT_JAR_PATH);
  private static final HashMap<String,Integer> Vocab = new HashMap<>();

  // Set of brands for labeling blocks
  static String[] brands = {"adidas", "bmw", "burger king", "coca cola", "esso", "heineken", "hp", "mcdonalds",
      "mercedes benz", "nvidia", "pepsi", "shell", "sri", "starbucks", "stella artois",
      "target", "texaco", "toyota", "twitter", "ups"};

  // Set of digits for labeling blocks
  static String[] digits = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
      "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"};

  // cardinal directions
  static String[] cardinal = {"NW", "N", "NE", "W", "TOP", "E", "SW", "S", "SE"};

  /**
   * Map every word which occurs at least twice to an integer
   */
  public static void computeVocabulary(ArrayList<Task> tasks) {
    // Compute the counts
    HashMap<String, Integer> counts = new HashMap<>();
    for (Task task : tasks) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String utterance : note.notes) {
            for (String word : tokenize(utterance)) {
              if (!counts.containsKey(word))
                counts.put(word, 0);
              counts.put(word, counts.get(word) + 1);
            }
          }
        }
      }
    }

    Vocab.put("<unk>",1);
    counts.keySet().stream().filter(w -> counts.get(w) > 1).forEachOrdered(w -> Vocab.put(w, Vocab.size() + 1));
    System.out.println(String.format("Created Vocabulary: %d of %d", Vocab.size(), counts.size()));
  }

  /**
   * Extract the block-id that moved
   * @param world_t
   * @param world_tp1
   * @return
   */
  public static int getSource(double[][] world_t, double[][] world_tp1) {
    for (int i = 0; i < world_t.length; ++i) {
      if (world_t[i][0] != world_tp1[i][0] || world_t[i][1] != world_tp1[i][1] || world_t[i][2] != world_tp1[i][2])
        return i;
    }
    System.err.println("We did not find a source block");
    return -1;
  }

  /**
   * Use the text to choose possible reference blocks
   */
  public static Set<Integer> getPossibleReferences(int source, String[] tokenized, String decoration) {
    // If random blank blocks then return the full set and choose the closest
    if (Configuration.blocktype.equals(Configuration.BlockType.Random)) {
      return new HashSet<>(Arrays.asList(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19));
    }
    String[] cleanup = {"the", "to", "on", "right", "line", "then", "even", "for", "them", "sit"};

    HashSet<String> words = new HashSet<>();
    words.addAll(Arrays.asList(tokenized));
    words.removeAll(Arrays.asList(cleanup));

    HashSet<Integer> blocks = new HashSet<>();
    if (decoration.equals("logo")) {
      ArrayList<String> brandparts = new ArrayList<>();
      String brand;
      // Try and find brands
      for (int brandid = 0 ; brandid < brands.length; ++brandid) {
        brand = brands[brandid];
        brandparts.clear();
        brandparts.addAll(Arrays.asList(Utils.whitespace_pattern.split(brand)));
        brandparts.add(brand.replace(" ", "-"));
        if (brand.equals("coca cola"))
          brandparts.add("coke");

        for (String part : brandparts) {
          for (String word : words) {
            if (part.equals(word) || (word.length() > 2  && StringUtils.editDistance(part, word) < 2))
              blocks.add(brandid);
          }
        }
      }
    } else if (decoration.equals("digit")) {
      // Are we using written digits (one, two, ...)
      HashSet<Integer> dblocks = new HashSet<>();
      String digit;
      for (int digitid = 0; digitid < digits.length; ++digitid) {
        digit = digits[digitid];
        for (String word : words) {
          if (StringUtils.editDistance(digit, word) < 2)
            dblocks.add(digitid);
        }
      }

      // Or numbers (1, 2, ...)
      HashSet<Integer> nblocks = new HashSet<>();
      for (int numeral = 1; numeral < 21; ++numeral) {
        if (words.contains(String.valueOf(numeral)) || words.contains(String.format("%dth", numeral)))
          nblocks.add(numeral - 1);
      }
      blocks = dblocks.size() > nblocks.size() ?  dblocks : nblocks;
    } else {
      return blocks;
    }

    if (blocks.contains(source))
      blocks.remove(source);
    return blocks;
  }

  /**
   * Return the closest reference block to the goal location (from the set returned by getPossibleReferences)
   */
  public static int getReference(int source, String[] tokenized, double[][] world, String decoration) {
    Set<Integer> blocks = getPossibleReferences(source, tokenized, decoration);
    if (!blocks.isEmpty()) {
      double maxD = 100000;
      double dist;
      int reference = -1;
      for (int block : blocks) {
        if (block < world.length && block != source) {
          dist = Utils.distance(world[block], world[source]);
          if (dist < maxD) {
            maxD = dist;
            reference = block;
          }
        }
      }
      return reference;
    } else {
      return -1;
    }
  }

  /**
   * Return Relative position of the source's new destination as compared to a reference block
   */
  public static int getDir(double[] source, double[] reference) {
    // Amended Ozan Dir scheme of Source relative to Reference
    //  1 2 3
    //  4 5 6
    //  7 8 9
    int dx = (int)Math.signum(source[0] - reference[0]);
    int dz = (int)Math.signum(source[2] - reference[2]);
    switch (dx) {
      case -1:
        switch (dz) {
          case -1:  // if dx < 0 and dz < 0 SW
            return 6;
          case 0:   // if dx < 0 and dz = 0 W
            return 3;
          case 1:   // if dx < 0 and dz > 0 NW
            return 0;
        }
      case 0:
        switch (dz) {
          case -1:    // if dx = 0 and dz < 0 S
            return 7;
          case 0:     // if dx = 0 and dz = 0 TOP
            return 4;
          case 1:     // if dx = 0 and dz > 0 N
            return 1;
        }
      case 1:
        switch (dz) {
          case -1:  // if dx > 0 and dz < 0 SE
            return 8;
          case 0:   // if dx > 0 and dz = 0 E
            return 5;
          case 1:   // if dx > 0 and dz > 0 NE
            return 2;
        }
    }
    return -1;
  }

  /**
   * Returns a flattened world representation (20x3) --> 60D vector
   */
  public static String getWorld(double[][] world) {
    double[] locs = new double[60];
    Arrays.fill(locs, -1);
    for (int i = 0; i < world.length; ++i) {
      locs[3*i] = world[i][0];
      locs[3*i + 1] = world[i][1];
      locs[3*i + 2] = world[i][2];
    }
    String toRet = "";
    for (double d : locs)
      toRet += String.format("%-5.2f ", d);
    return toRet;
  }

  /**
   * Returns a tokenized string[]
   */
  public static String[] tokenize(String utterance) {
    String[] tagged = Utils.whitespace_pattern.split(tagger.tagString(utterance.replace(",", " , ")));
    for (int i = 0; i < tagged.length; ++i)
      tagged[i] = Utils.dash_pattern.split(tagged[i])[0].toLowerCase();
    return tagged;
  }

  /**
   * Convert utterance to a sparse vector.  Words are UNKed according to Vocab dictionary
   */
  public static String unkUtterance(String[] tokenized) {
    String toRet = "";
    for (String word : tokenized) {
      if (!Vocab.containsKey(word))
        toRet += "1  ";
      else
        toRet += String.format("%-2d ", Vocab.get(word));
    }
    return toRet;
  }

  /**
   * Takes a series of tasks and extracts prediction and conditioning information based on Configuration file.
   * Results is a sparse matrix (ints) or floats
   */
  public static void createMatrix(ArrayList<Task> data, String filename) throws IOException {
    BufferedWriter BW = TextFile.Writer(filename);
    BufferedWriter Human = TextFile.Writer(filename + ".human");
    for (Task task : data) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String utterance : note.notes) {
            // Compute predictions
            int source = -1, reference = -1, Dir = -1;
            for (Information info : Configuration.predict) {
              switch (info) {
                case Source:
                  source = getSource(task.states[note.start], task.states[note.finish]);
                  BW.write(String.format(" %d ", source));
                  Human.write(String.format(" %-12s ", task.decoration.equals("logo") ? brands[source] : digits[source]));
                  break;
                case Reference:
                  reference = getReference(source, tokenize(utterance), task.states[note.finish], task.decoration);
                  int t_reference = reference > -1 ? reference : source;
                  BW.write(String.format(" %d ", t_reference));
                  Human.write(String.format(" %-15s ", t_reference > -1 ? task.decoration.equals("logo") ? brands[t_reference] : digits[t_reference] : "NULL"));
                  break;
                case Direction:
                  if (reference != -1)
                    Dir = getDir(task.states[note.finish][source], task.states[note.finish][reference]);
                  else
                    Dir = getDir(task.states[note.finish][source], task.states[note.start][source]);
                  BW.write(String.format(" %d ", Dir));
                  Human.write(String.format(" %-3s ", cardinal[Dir]));
                  break;
                case XYZ:
                  BW.write(String.format(" %-5.2f %-5.2f %-5.2f ", task.states[note.finish][source][0],
                      task.states[note.finish][source][1], task.states[note.finish][source][2]));
                  Human.write(String.format(" %-5.2f %-5.2f %-5.2f ", task.states[note.finish][source][0],
                      task.states[note.finish][source][1], task.states[note.finish][source][2]));
                  break;
                default:
                  System.err.println("We don't predict " + info);
                  return;
              }
            }
            // Compute conditioning variables
            for (Information info : Configuration.condition) {
              switch (info) {
                case CurrentWorld:
                  BW.write(getWorld(task.states[note.start]));
                  Human.write(String.format(" %s ", task.images[note.start]));
                  break;
                case NextWorld:
                  BW.write(getWorld(task.states[note.finish]));
                  Human.write(String.format(" %s ", task.images[note.finish]));
                  break;
                case Utterance:
                  BW.write(unkUtterance(tokenize(utterance)));
                  Human.write("    " + utterance);
                  break;
                default:
                  System.err.println("We don't condition on " + info);
                  return;
              }
            }
            BW.write("\n");
            Human.write("\n");
          }
        }
      }
    }
    BW.close();
    Human.close();
    System.out.println("Created " + filename);
  }

  /**
   * Currently we assume all data is STDir
   */
  public static void createRecords(ArrayList<Task> data, String filename) throws IOException {
    BufferedWriter BW = TextFile.Writer(filename);
    int idx = 0;
    String[] utterance;
    String[][] tokens = new String[9][];
    for (Task task : data) {
      if (task.decoration.equals("logo")) {
        for (Note note : task.notes) {
          if (note.type.equals("A0")) {
            for (int i = 0; i < note.notes.length; ++i) {
              tokens[i] = tokenize(note.notes[i]);
            }
            for (int i = 0; i < note.notes.length; ++i) {
              utterance = tokens[i];
              int source = getSource(task.states[note.start], task.states[note.finish]);
              int reference = getReference(source, utterance, task.states[note.finish], task.decoration);
              int Dir;
              if (reference != -1)
                Dir = getDir(task.states[note.finish][source], task.states[note.finish][reference]);
              else
                Dir = getDir(task.states[note.finish][source], task.states[note.start][source]);
              BW.write(String.format("Example_%d\n", ++idx));
              BW.write(Arrays.asList(utterance).stream().collect(Collectors.joining(" ")) + "\n");
              for (int j = 0; j < note.notes.length; ++j) {
                if (i != j)
                  BW.write(Arrays.asList(note.notes[j]).stream().collect(Collectors.joining(" ")) + "\n");
              }
              BW.write(String.format(".id:0\t.type:source\t@block:%d\n", source));
              BW.write(String.format(".id:1\t.type:reference\t@block:%s\n", reference > -1 ? String.valueOf(reference) : "--"));
              BW.write(String.format(".id:2\t.type:pos\t@Dir:%d\n", Dir));
              BW.write("0 0 1 2\n");
            }
          }
        }
      }
    }
    BW.close();
    System.out.println("Created " + filename);
  }


  public static strictfp void main(String[] args) throws Exception {
    Configuration.setConfiguration(args.length > 0 ? args[0] : "JSONReader/config.properties");

    ArrayList<Task> Train = LoadJSON.readJSON(Configuration.training);
    ArrayList<Task> Test = LoadJSON.readJSON(Configuration.testing);
    ArrayList<Task> Dev = LoadJSON.readJSON(Configuration.development);

    switch(Configuration.output) {
      case Matrix:
        computeVocabulary(Train);
        createMatrix(Train, "Train.mat");
        createMatrix(Test, "Test.mat");
        createMatrix(Dev, "Dev.mat");
        break;
      case Records:
        createRecords(Train, "Train.records");
        createRecords(Test, "Test.records");
        createRecords(Dev, "Dev.records");
        break;
    }
  }
}
