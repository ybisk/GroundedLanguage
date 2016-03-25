import JSON.Note;
import JSON.Task;
import com.google.gson.Gson;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.StringUtils;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.regex.Pattern;

public class LoadJSON {
  private static final Gson gsonReader = new Gson();
  private static final Pattern whitespace_pattern = Pattern.compile("\\s+");
  private static final Pattern dash_pattern = Pattern.compile("_");
  private static MaxentTagger tagger = new MaxentTagger(MaxentTagger.DEFAULT_JAR_PATH);
  private static final HashMap<String,Integer> Vocab = new HashMap<>();

  public static ArrayList<Task> readJSON(String filename) throws IOException {
    BufferedReader BR = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(new File(filename))), "UTF-8"));
    ArrayList<Task> Tasks = new ArrayList<>();
    String line;
    while ((line = BR.readLine()) != null)
      Tasks.add(gsonReader.fromJson(line, Task.class));
    return Tasks;
  }

  public static String UPOS(String PTB) {
    switch(PTB) {
      case "!"      : return ".";
      case "#"      : return ".";
      case "$"      : return ".";
      case "''"     : return ".";
      case "("      : return ".";
      case ")"      : return ".";
      case ","      : return ".";
      case "-LRB-"  : return ".";
      case "-RRB-"  : return ".";
      case "."      : return ".";
      case ":"      : return ".";
      case "?"      : return ".";
      case "CC"     : return "CONJ";
      case "CD"     : return "NUM";
      case "DT"     : return "DET";
      case "EX"     : return "DET";
      case "FW"     : return "X";
      case "IN"     : return "ADP";
      case "JJ"     : return "ADJ";
      case "JJR"    : return "ADJ";
      case "JJRJR"  : return "ADJ";
      case "JJS"    : return "ADJ";
      case "LS"     : return "X";
      case "MD"     : return "VERB";
      case "NN"     : return "NOUN";
      case "NNP"    : return "NOUN";
      case "NNPS"   : return "NOUN";
      case "NNS"    : return "NOUN";
      case "NP"     : return "NOUN";
      case "PDT"    : return "DET";
      case "POS"    : return "PRT";
      case "PRP"    : return "PRON";
      case "PRP$"   : return "PRON";
      case "PRT"    : return "PRT";
      case "RB"     : return "ADV";
      case "RBR"    : return "ADV";
      case "RBS"    : return "ADV";
      case "RN"     : return "X";
      case "RP"     : return "PRT";
      case "SYM"    : return "X";
      case "TO"     : return "PRT";
      case "UH"     : return "X";
      case "VB"     : return "VERB";
      case "VBD"    : return "VERB";
      case "VBG"    : return "VERB";
      case "VBN"    : return "VERB";
      case "VBP"    : return "VERB";
      case "VBZ"    : return "VERB";
      case "VP"     : return "VERB";
      case "WDT"    : return "DET";
      case "WH"     : return "X";
      case "WP"     : return "PRON";
      case "WP$"    : return "PRON";
      case "WRB"    : return "ADV";
      case "``"     : return " .";
      default: return "BLAH";
    }
  }

  /**
   * Compute most frequent words by POS tag
   */
  public static void statistics(ArrayList<Task> tasks) throws IOException {
    BufferedWriter Sents = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(
        new FileOutputStream(new File("Sentences.txt.gz"))), "UTF-8"));
    HashMap<String,HashMap<String,Integer>> tagTokenMap = new HashMap<>();
    HashMap<String,Integer> tokenMap;
    String taggedString;
    String[] tokens;
    String[] taggedTokens;
    for (Task task : tasks) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String utterance : note.notes) {
            taggedString = tagger.tagString(utterance.replace(","," , "));
            Sents.write(taggedString + "\n");
            tokens = taggedString.split("\\s+");
            for (String tok : tokens) {
              taggedTokens = tok.split("_");
              String tag = taggedTokens[1]; //UPOS(taggedTokens[1]);
              tokenMap = tagTokenMap.get(tag);
              if (tokenMap == null) {
                tagTokenMap.put(tag, new HashMap<>());
                tokenMap = tagTokenMap.get(tag);
              }
              String word = taggedTokens[0].toLowerCase();
              if (!tokenMap.containsKey(word))
                tokenMap.put(word, 0);
              tokenMap.put(word, tokenMap.get(word) + 1);
            }
          }
        }
      }
    }
    Sents.close();

    ArrayList<Tuple> tuples = new ArrayList<>();
    for (String tag : tagTokenMap.keySet()) {
      tuples.add(new Tuple(tag, tagTokenMap.get(tag).size()));
    }
    Collections.sort(tuples);

    BufferedWriter BW = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(
        new FileOutputStream(new File("TypeDistribution.txt.gz"))), "UTF-8"));
    for (Tuple tuple : tuples) {
      BW.write(String.format("%-5d  %-5s\n", (int)tuple.value(), tuple.content()));
      ArrayList<Tuple> words = new ArrayList<>();
      for (String word : tagTokenMap.get(tuple.content()).keySet()) {
        words.add(new Tuple(word, tagTokenMap.get(tuple.content()).get(word)));
      }
      Collections.sort(words);
      for (int i = 0; i < words.size(); ++i)
        if (words.get(i).value() > 1)
          BW.write(String.format("   %-4d %-8s\n", (int)words.get(i).value(), words.get(i).content()));
      BW.write("\n");
    }
    BW.close();
  }

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

  public static int getSource(double[][] world_t, double[][] world_tp1) {
    for (int i = 0; i < world_t.length; ++i) {
      if (world_t[i][0] != world_tp1[i][0] || world_t[i][1] != world_tp1[i][1] || world_t[i][2] != world_tp1[i][2])
        return i;
    }
    System.err.println("We did not find a source block");
    return -1;
  }

  public static double distance(double[] A, double[] B) {
    return Math.sqrt(Math.pow(A[0] - B[0], 2) + Math.pow(A[1] - B[1], 2) + Math.pow(A[2] - B[2], 2)) / 0.1524;
  }

  public static Set<Integer> getPossibleTargets(int source, String[] tokenized, String decoration) {
    // Set of brands for labeling blocks
    String[] brands = {"adidas", "bmw", "burger king", "coca cola", "esso", "heineken", "hp", "mcdonalds",
        "mercedes benz", "nvidia", "pepsi", "shell", "sri", "starbucks", "stella artois",
        "target", "texaco", "toyota", "twitter", "ups"};

    // Set of digits for labeling blocks
    String[] digits = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"};

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
        brandparts.addAll(Arrays.asList(whitespace_pattern.split(brand)));
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

  public static int getTarget(int source, String[] tokenized, double[][] world, String decoration) {
    Set<Integer> blocks = getPossibleTargets(source, tokenized, decoration);
    if (!blocks.isEmpty()) {
      double maxD = 100000;
      double dist;
      int target = -1;
      for (int block : blocks) {
        if (block < world.length) {
          dist = distance(world[block], world[source]);
          if (dist < maxD) {
            maxD = dist;
            target = block;
          }
        }
      }
      return target;
    } else {
      return source;
    }
  }

  public static int getRP(double[] source, double[] target) {
    // Amended Ozan RP scheme of Source relative to Target
    //  1 2 3
    //  4 5 6
    //  7 8 9
    int dx = (int)Math.signum(source[0] - target[0]);
    int dz = (int)Math.signum(source[2] - target[2]);
    switch (dx) {
      case -1:
        switch (dz) {
          case -1:  // if dx < 0 and dz < 0 SW
            return 7;
          case 0:   // if dx < 0 and dz = 0 W
            return 4;
          case 1:   // if dx < 0 and dz > 0 NW
            return 1;
        }
      case 1:
        switch (dz) {
          case -1:  // if dx > 0 and dz < 0 SE
            return 9;
          case 0:   // if dx > 0 and dz = 0 E
            return 6;
          case 1:   // if dx > 0 and dz > 0 NE
            return 3;
        }
      case 0:
        switch (dz) {
        case -1:    // if dx = 0 and dz < 0 S
          return 8;
        case 0:     // if dx = 0 and dz = 0 TOP
          return 5;
        case 1:     // if dx = 0 and dz > 0 N
          return 2;
      }
    }
    return -1;
  }

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

  public static String[] tokenize(String utterance) {
    String[] tagged = whitespace_pattern.split(tagger.tagString(utterance.replace(",", " , ")));
    for (int i = 0; i < tagged.length; ++i)
      tagged[i] = dash_pattern.split(tagged[i])[0].toLowerCase();
    return tagged;
  }

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

  public static void createMatrix(ArrayList<Task> data, Information[] condition, Information[] predict, BufferedWriter BW) throws IOException {
    for (Task task : data) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String utterance : note.notes) {
            // Compute predictions
            int source = -1, target = -1;
            for (Information info : predict) {
              switch (info) {
                case Source:
                  source = getSource(task.states[note.start], task.states[note.finish]);
                  BW.write(String.format(" %d ", source));
                  break;
                case Target:
                  target = getTarget(source, tokenize(utterance), task.states[note.finish], task.decoration);
                  BW.write(String.format(" %d ", target));
                  break;
                case RelativePosition:
                  if (target != -1)
                    BW.write(String.format(" %d ", getRP(task.states[note.finish][source], task.states[note.finish][target])));
                  else
                    BW.write(String.format(" %d ", getRP(task.states[note.finish][source], task.states[note.start][source])));
                  break;
                case XYZ:
                  BW.write(String.format(" %-5.2f %-5.2f %-5.2f ", task.states[note.finish][source][0],
                      task.states[note.finish][source][1], task.states[note.finish][source][2]));
                default:
                  System.err.println("We don't predict " + info);
                  return;
              }
            }
            // Compute conditioning variables
            for (Information info : condition) {
              switch (info) {
                case CurrentWorld:
                  BW.write(getWorld(task.states[note.start]));
                  break;
                case NextWorld:
                  BW.write(getWorld(task.states[note.finish]));
                  break;
                case Utterance:
                  BW.write(unkUtterance(tokenize(utterance)));
                  break;
                default:
                  System.err.println("We don't condition on " + info);
                  return;
              }
            }
            BW.write("\n");
          }
        }
      }
    }
  }

  public static strictfp void main(String[] args) throws Exception {
    ArrayList<Task> Train = readJSON(args[0]);
    System.out.print("Read Training ");
    ArrayList<Task> Test = readJSON(args[1]);
    System.out.print(" Testing");
    ArrayList<Task> Dev = readJSON(args[2]);
    System.out.println(" Development");
    //statistics(Train);

    computeVocabulary(Train);

    // Settings
    Information[] condition = new Information[] {Information.CurrentWorld, Information.Utterance};
    Information[] predict = new Information[] {Information.Source, Information.Target, Information.RelativePosition};

    createMatrix(Train, condition, predict, MatrixFile.Writer("Train.mat"));
    System.out.println("Created Train.mat");
    createMatrix(Test, condition, predict, MatrixFile.Writer("Test.mat"));
    System.out.println("Created Test.mat");
    createMatrix(Dev, condition, predict, MatrixFile.Writer("Dev.mat"));
    System.out.println("Created Dev.mat");
  }

  public enum Information {
    CurrentWorld, Utterance, NextWorld, Source, Target, RelativePosition, XYZ
  }
}
