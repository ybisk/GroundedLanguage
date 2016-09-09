import JSON.Note;
import JSON.Task;
import com.google.gson.Gson;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.StringUtils;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.regex.Pattern;

public class LoadJSON {
  static final Gson gsonReader = new Gson();

  /**
   * Deserialize the JSONs
   */
  public static ArrayList<Task> readJSON(String filename) throws IOException {
    BufferedReader BR = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(new File(filename))), "UTF-8"));
    ArrayList<Task> Tasks = new ArrayList<>();
    String line;
    while ((line = BR.readLine()) != null)
      Tasks.add(gsonReader.fromJson(line, Task.class));

    System.out.println("Read " + filename);
    return Tasks;
  }

  /**
   * Compute most frequent words by POS tag
   */
  public static void POSStatistics(ArrayList<Task> tasks) throws IOException {
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
            taggedString = CreateTrainingData.tagger.tagString(utterance.replace(","," , "));
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

  private static int binLength(int length) {
    if (length <= 5)
      return 5;
    if (length <= 10)
      return 10;
    if (length <= 20)
      return 20;
    if (length <= 40)
      return 40;
    if (length <= 80)
      return 80;
    return -1;
  }
  /**
   * Compute Type and Token counts.
   * Compute length distribution
   */
  public static HashMap<String,Integer> CorpusStatistics(ArrayList<Task> tasks, String type) throws IOException {
    HashSet<String> tokenSet = new HashSet<>();
    int tokenCount = 0;
    ArrayList<Integer> tokenCounts = new ArrayList<>();
    HashMap<String,Integer> counts = new HashMap<>();
    int utterances = 0;
    int bin;
    HashMap<Integer,Integer> Lengths = new HashMap<>();
    String taggedString;
    String[] tokens;
    String[] taggedTokens;
    for (Task task : tasks) {
      for (Note note : task.notes) {
        if (note.type.equals(type)) {
          for (String utterance : note.notes) {
            taggedString = CreateTrainingData.tagger.tagString(utterance.replace(","," , ").replace("-"," - ").replace("/"," / "));
            tokens = taggedString.split("\\s+");
            for (String tok : tokens) {
              taggedTokens = tok.split("_");
              String word = taggedTokens[0].toLowerCase();
              tokenSet.add(word);
              if (!counts.containsKey(word))
                counts.put(word, 0);
              counts.put(word, counts.get(word) + 1);
              tokenCount += 1;
            }
            bin = binLength(tokens.length);
            if (!Lengths.containsKey(bin))
              Lengths.put(bin, 0);
            Lengths.put(bin, Lengths.get(bin) + 1);
            ++utterances;
            tokenCounts.add(tokens.length);
          }
        }
      }
    }
    System.out.println("Type: " + type);
    System.out.println("Types: " + tokenSet.size());
    System.out.println("Tokens: " + tokenCount);
    ArrayList<Integer> keys = new ArrayList<>(Lengths.keySet());
    Collections.sort(keys);
    System.out.println("Lengths:");
    for (int i : keys)
      System.out.println(String.format("    %-2d  %-6d", i,  Lengths.get(i)));
    System.out.println("Utterances: " + utterances);
    double mean = 1.0*tokenCount/utterances;
    System.out.println("Average Length: " + mean);
    double std = 0.0;
    for (Integer length : tokenCounts)
      std += Math.pow(length - mean, 2);
    System.out.println("Std Dev: " + Math.sqrt(std/utterances));
    return counts;
  }


  /**
   * Compute percentage of dataset written by any given person
   */
  public static void authorStatistics(ArrayList<Task> tasks) throws IOException {
    HashMap<String,Integer> authors = new HashMap<>();
    double total = 0;
    for (Task task : tasks) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String user : note.users) {
            if (!authors.containsKey(user))
              authors.put(user, 0);
            authors.put(user, authors.get(user) + 1);
            ++total;
          }
        }
      }
    }
    ArrayList<Tuple> tuples = new ArrayList<>();
    for (String user : authors.keySet())
      tuples.add(new Tuple(user, authors.get(user)));
    Collections.sort(tuples);

    System.out.println("Total annotators: " + tuples.size());
    for (Tuple tuple : tuples)
      System.out.println(String.format("%-25s %d %f", tuple.content(), (int) tuple.value(), 100.0*tuple.value()/total));
  }


  public static strictfp void main(String[] args) throws Exception {
    Configuration.setConfiguration(args.length > 0 ? args[0] : "JSONReader/config.properties");

    //System.out.println("Train");
    //CorpusStatistics(readJSON(Configuration.training), "A0");
    //System.out.println("Test");
    //CorpusStatistics(readJSON(Configuration.testing), "A0");
    //System.out.println("Dev");
    //CorpusStatistics(readJSON(Configuration.development), "A0");

    System.out.println("All");
    ArrayList<Task> all = readJSON(Configuration.training);
    //all.addAll(readJSON(Configuration.testing));
    all.addAll(readJSON(Configuration.development));
    HashMap<String,Integer> A0 = CorpusStatistics(all, "A0");
    HashMap<String,Integer> A1 = CorpusStatistics(all, "A1");
    HashMap<String,Integer> A2 = CorpusStatistics(all, "A2");

    String setting = "Blank";
    BufferedWriter writer = TextFile.Writer(setting + ".A0.freq");
    ArrayList<Tuple> tuples = new ArrayList<>();
    for (String word : A0.keySet())
      tuples.add(new Tuple(word, A0.get(word)));
    Collections.sort(tuples);
    for (Tuple T : tuples)
      writer.write(String.format("%-30s %d\n", T.content(), (int) T.value()));
    writer.close();

    writer = TextFile.Writer(setting + ".A1.freq");
    tuples = new ArrayList<>();
    for (String word : A1.keySet())
      tuples.add(new Tuple(word, A1.get(word)));
    Collections.sort(tuples);
    for (Tuple T : tuples)
      writer.write(String.format("%-30s %d\n", T.content(), (int) T.value()));
    writer.close();

    writer = TextFile.Writer(setting + ".A2.freq");
    tuples = new ArrayList<>();
    for (String word : A2.keySet())
      tuples.add(new Tuple(word, A2.get(word)));
    Collections.sort(tuples);
    for (Tuple T : tuples)
      writer.write(String.format("%-30s %d\n", T.content(), (int) T.value()));
    writer.close();



    writer = TextFile.Writer(setting + ".A0.txt");
    for (Task task : all) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String utter : note.notes)
            writer.write(utter + "\n");
        }
      }
    }
    writer.close();
    writer = TextFile.Writer(setting + ".A1.txt");
    for (Task task : all) {
      for (Note note : task.notes) {
        if (note.type.equals("A1")) {
          for (String utter : note.notes)
            writer.write(utter + "\n");
        }
      }
    }
    writer.close();
    writer = TextFile.Writer(setting + ".A2.txt");
    for (Task task : all) {
      for (Note note : task.notes) {
        if (note.type.equals("A2")) {
          for (String utter : note.notes)
            writer.write(utter + "\n");
        }
      }
    }
    writer.close();



    //Set<String> Keys = new HashSet<>(A0.keySet());
    //tuples = new ArrayList<>();
    //for (String word : Keys)
    //  tuples.add(new Tuple(word, A0.get(word)));
    //Collections.sort(tuples);
    //System.out.println(tuples);

    //Keys.clear();
    //tuples.clear();
    //Keys = new HashSet<>(A1.keySet());
    //Keys.removeAll(A0.keySet());
    //tuples = new ArrayList<>();
    //for (String word : Keys)
    //  tuples.add(new Tuple(word, A1.get(word)));
    //Collections.sort(tuples);
    //System.out.println(tuples);

    //Keys.clear();
    //tuples.clear();
    //Keys = new HashSet<>(A2.keySet());
    //Keys.removeAll(A1.keySet());
    //Keys.removeAll(A0.keySet());
    //for (String word : Keys)
    //  tuples.add(new Tuple(word, A2.get(word)));
    //Collections.sort(tuples);
    //System.out.println(tuples);
  }
}
