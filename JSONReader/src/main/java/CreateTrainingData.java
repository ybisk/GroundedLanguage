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
  private static final HashMap<String,Integer> counts = new HashMap<>();

  // Set of brands for labeling blocks
  static List<String> brands = Arrays.asList("adidas", "bmw", "burger king", "coca cola", "esso", "heineken", "hp",
      "mcdonalds", "mercedes benz", "nvidia", "pepsi", "shell", "sri", "starbucks", "stella artois",
      "target", "texaco", "toyota", "twitter", "ups");

  // Set of digits for labeling blocks
  static List<String> digits = Arrays.asList("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
      "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty");

  // cardinal directions
  static String[] cardinal = {"NW", "N", "NE", "W", "TOP", "E", "SW", "S", "SE"};

  // Generation types
  static String[] generationTypes = {"Solo", "SRD", "Line", "Axes"};

  static List<String> boring = Arrays.asList("block");

  // Stop words:  NLTK
  static Set<String> stopwords = new HashSet<> (Arrays.asList(
      "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "yo", "your", "yours",
      "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
      "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
      "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
      "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
      "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
      "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
      "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
      "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
      "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
      "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
      "block", ".", ",", "Place", "Put", "Position","blocks", "number", "Place", "Position", "place"));

  /**
   * Return best fit or -1
   */
  private static int isaName(String decoration, String word) {
    if (decoration.equals("logo")) {
      ArrayList<String> brandparts = new ArrayList<>();
      String brand;
      // Try and find brands
      for (int brandid = 0 ; brandid < brands.size(); ++brandid) {
        brand = brands.get(brandid);
        brandparts.clear();
        brandparts.addAll(Arrays.asList(Utils.whitespace_pattern.split(brand)));
        brandparts.add(brand.replace(" ", "-"));
        if (brand.equals("coca cola"))
          brandparts.add("coke");

        for (String part : brandparts) {
          if (part.equals(word) || (word.length() > 2  && StringUtils.editDistance(part, word) < 2))
            return brandid;
        }
      }
    } else if (decoration.equals("digit")) {
      // Are we using written digits (one, two, ...)
      String digit;
      for (int digitid = 0; digitid < digits.size(); ++digitid) {
        digit = digits.get(digitid);
        if (StringUtils.editDistance(digit, word) < 2)
          return digitid;
      }

      // Or numbers (1, 2, ...)
      for (int numeral = 1; numeral < 21; ++numeral) {
        if (word.contains(String.valueOf(numeral)) || word.contains(String.format("%dth", numeral)))
          return numeral - 1;
      }
    }
    return -1;
  }

  /**
   * Map every word which occurs at least twice to an integer
   */
  public static void computeVocabulary(ArrayList<Task> tasks)  throws IOException {
    // Compute the counts
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

    Vocab.put("<s>",1);
    Vocab.put("</s>",2);
    Vocab.put("<unk>",3);
    counts.keySet().stream().filter(w -> counts.get(w) > 1).forEachOrdered(w -> Vocab.put(w, Vocab.size() + 1));
    System.out.println(String.format("Created Vocabulary: %d of %d", Vocab.size(), counts.size()));
    
    BufferedWriter BW = TextFile.Writer("Vocabulary.txt");
    for (String word : Vocab.keySet())
      BW.write(String.format("%s %d\n", word, Vocab.get(word)));
    BW.close();
  }

  /**
   * Extract the block-id that moved
   */
  public static int getSource(int start, int finish, Task task) {
    double[][] world_t = task.states[start];
    double[][] world_tp1 = task.states[finish];
    int id = -1;
    double dist;
    double max_dist = 0;
    // Who physically moved furthest
    for (int i = 0; i < world_t.length; ++i) {
      dist = Utils.distance(world_t[i], world_tp1[i]);
      if (dist > max_dist) {
        max_dist = dist;
        id = i;
      }
    }

    if (id != -1 || task.orientations == null)
      return id;

    // Who rotated the most?!?
    double[][] orientation_tp = task.orientations[start];
    double[][] orientation_tp1 = task.orientations[finish];
    max_dist = 0;
    for (int i = 0; i < orientation_tp.length; ++i) {
      dist = Utils.distance(orientation_tp[i], orientation_tp1[i]);
      if (dist > max_dist) {
        max_dist = dist;
        id = i;
      }
    }

    if (id != -1)
      return id;

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

    HashSet<Integer> blocks;
    if (decoration.equals("logo") || decoration.equals("digit")) {
      blocks = words.stream().map(w -> isaName(decoration, w)).filter(i -> i != -1).collect(Collectors.toCollection(HashSet::new));
    } else {
      return new HashSet<>();
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
    if (Math.abs(source[1] - reference[1]) > Math.max(Math.abs(source[0] - reference[0]),Math.abs(source[2] - reference[2])))
      return 4;
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

  private static HashSet<String> tagsToSkip = new HashSet<>(Arrays.asList("CC", "DT", "MD", "POS", "PRP", "PRP$",
      "VB", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"));
  public static String[] tokenize(String utterance) { return tokenize(utterance, true); }
  /**
   * Returns a tokenized string[]
   */
  public static String[] tokenize(String utterance, boolean keepEverything) {
    String[] tagged = Utils.whitespace_pattern.split(tagger.tagString(utterance.replace(",", " , ")));
    ArrayList<String> pruned = new ArrayList<>();
    String[] split;
    for (int i = 0; i < tagged.length; ++i) {
      split = Utils.dash_pattern.split(tagged[i]);
      if (keepEverything || !tagsToSkip.contains(split[1]))
        pruned.add(split[0]);
    }
    return pruned.toArray(new String[pruned.size()]);
  }

  /**
   * Remove stopwords and proper nouns from the utterance
   */
  public static String[] FreqWords(String decoration, String[] utterance) {
    ArrayList<String> words = Arrays.stream(utterance)
        .filter(w -> !stopwords.contains(w.toLowerCase()))    // Remove stop words
        .filter(w -> isaName(decoration, w) == -1)            // Remove proper nouns
        .collect(Collectors.toCollection(ArrayList::new));
    words.sort(Comparator.naturalOrder());                    // Sort for consistency
    return words.toArray(new String[words.size()]);
  }

  /**
   * Use the language of the utterance to determine if the annotator is using one of four descriptions:
   * 1) Source, Reference, Direction
   * 2) Direction + Distance
   * 3) Lines
   * 4) Axes
   */
  public static int generationType(int source, String[] utterance, String decoration) {
    Set<Integer> blocks = getPossibleReferences(source, utterance, decoration);
    HashSet<String> words = new HashSet<>();
    words.addAll(Arrays.asList(utterance));
    // If there are no references, it is solo
    if (blocks.size() == 0)       // 0: Solo
      return 0;
    else if (blocks.size() == 1)  // 1: SRD
      return 1;
    else {
      if (words.contains("line") || words.contains("between"))
        return 2;                 // 2: Line
      else {
        return 3;                 // 3: Axes
      }
    }
  }

  /**
   * Convert utterance to a sparse vector.  Words are UNKed according to Vocab dictionary
   */
  public static String unkUtterance(String[] tokenized) {
    String toRet = "1 ";
    for (String word : tokenized) {
      if (!Vocab.containsKey(word))
        toRet += "3  ";
      else
        toRet += String.format("%-2d ", Vocab.get(word));
    }
    return toRet + " 2 ";
  }

  public static void appendInformation(Information info, Task task, Note note, String utterance, BufferedWriter BW,
                                       BufferedWriter Human) throws IOException {
    // Several pieces require knowledge of the source and reference
    int source = -1;
    if (info == Information.Source || info == Information.Reference || info == Information.Direction ||
        info == Information.tXYZ   || info == Information.sXYZ || info == Information.GenerationType ||
        info == Information.KeyWords) {
      if (note.finish == task.states.length)
        System.err.println("Well well well..." + note.finish + " \t" + task.states.length);
      source = getSource(note.start, note.finish, task);
    }
    int reference = -1;
    if (info == Information.Reference || info == Information.Direction || info == Information.KeyWords)
      reference = getReference(source, tokenize(utterance), task.states[note.finish], task.decoration);

    int Dir;
    switch (info) {
      case Source:
        BW.write(String.format(" %d ", source));
        Human.write(String.format(" %-12s ", task.decoration.equals("logo") ? brands.get(source) : digits.get(source)));
        break;
      case Reference:
        int t_reference = reference > -1 ? reference : source;
        BW.write(String.format(" %d ", t_reference));
        Human.write(String.format(" %-15s ", t_reference > -1 ? task.decoration.equals("logo") ? brands.get(t_reference) : digits.get(t_reference) : "NULL"));
        break;
      case Direction:
        if (reference != -1)
          Dir = getDir(task.states[note.finish][source], task.states[note.finish][reference]);
        else
          Dir = getDir(task.states[note.finish][source], task.states[note.start][source]);
        BW.write(String.format(" %d ", Dir));
        Human.write(String.format(" %-3s ", cardinal[Dir]));
        break;
      case tXYZ:
        BW.write(String.format(" %-5.2f %-5.2f %-5.2f ", task.states[note.finish][source][0],
            task.states[note.finish][source][1], task.states[note.finish][source][2]));
        Human.write(String.format(" %-5.2f %-5.2f %-5.2f ", task.states[note.finish][source][0],
            task.states[note.finish][source][1], task.states[note.finish][source][2]));
        break;
      case sXYZ:
        BW.write(String.format(" %-5.2f %-5.2f %-5.2f ", task.states[note.start][source][0],
            task.states[note.start][source][1], task.states[note.start][source][2]));
        Human.write(String.format(" %-5.2f %-5.2f %-5.2f ", task.states[note.start][source][0],
            task.states[note.start][source][1], task.states[note.start][source][2]));
        break;
      case CurrentWorld:
        BW.write(getWorld(task.states[note.start]));
        Human.write(String.format(" %s ", task.images[note.start]));
        break;
      case NextWorld:
        BW.write(getWorld(task.states[note.finish]));
        Human.write(String.format(" %s ", task.images[note.finish]));
        break;
      case KeyWords:
        if (reference != -1)
          Dir = getDir(task.states[note.finish][source], task.states[note.finish][reference]);
        else
          Dir = getDir(task.states[note.finish][source], task.states[note.start][source]);
        utterance += " " + cardinal[Dir];
        BW.write(unkUtterance(FreqWords(task.decoration, tokenize(utterance))));
        Human.write("    " + Arrays.asList(FreqWords(task.decoration, tokenize(utterance, false))).stream().collect(Collectors.joining(" ")));
        break;
      case Utterance:
        BW.write(unkUtterance(tokenize(utterance)));
        Human.write("    " + utterance);
        break;
      case GenerationType:
        int g = generationType(source, tokenize(utterance), task.decoration);
        BW.write(String.format(" %d ", g));
        Human.write(String.format(" %s ", generationTypes[g]));
        break;
      default:
        System.err.println("We don't handle " + info);
        return;
    }
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
            for (Information info : Configuration.predict) {
              appendInformation(info, task, note, utterance, BW, Human);
            }
            // Compute conditioning variables
            for (Information info : Configuration.condition) {
              appendInformation(info, task, note, utterance, BW, Human);
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
              int source = getSource(note.start, note.finish, task);
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
    Configuration.setConfiguration(args.length > 0 ? args[0] : "config.properties");

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
