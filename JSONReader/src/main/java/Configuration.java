import java.util.List;

public class Configuration {

  // Default Parameter Values
  public static String training = "BlockWorld/DataCreation/trainset/trainset.json.gz";
  public static String testing = "BlockWorld/DataCreation/testset/testset.json.gz";
  public static String development = "BlockWorld/DataCreation/devset/devset.json.gz";

  // What to extract
  public static Information[] condition = new Information[] {Information.CurrentWorld, Information.Utterance};
  public static Information[] predict = new Information[] {Information.Source, Information.Reference, Information.Direction};

  // What the output format is
  public static OutputFormat output = OutputFormat.Matrix;

  public static void setConfiguration(String configuration) {
    List<String> config = TextFile.Read(configuration);
    String[] split;
    for (String line: config) {
      split = line.split("=");
      if (line.length() > 0 && !split[0].startsWith("#")) {
        split[0] = split[0].trim();
        switch (split[0]) {
          case "training":
            training = split[1].trim();
            break;
          case "testing":
            testing = split[1].trim();
            break;
          case "development":
            development = split[1].trim();
            break;
          case "condition":
            split = LoadJSON.whitespace_pattern.split(split[1].trim());
            condition = new Information[split.length];
            for (int i = 0; i < split.length; ++i) {
              condition[i] = Information.valueOf(split[i].trim());
            }
            break;
          case "predict":
            split = LoadJSON.whitespace_pattern.split(split[1].trim());
            predict = new Information[split.length];
            for (int i = 0; i < split.length; ++i) {
              predict[i] = Information.valueOf(split[i].trim());
            }
            break;
          case "output":
            output = OutputFormat.valueOf(split[1].trim());
            break;
          default:
            System.err.println("Invalid configuration option: " + split[0] + " ... ignoring");
        }
      }
    }
  }

  public enum OutputFormat {
    Matrix, Records
  }
}
