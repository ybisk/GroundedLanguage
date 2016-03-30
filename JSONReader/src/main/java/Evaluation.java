import JSON.Note;
import JSON.Task;

import java.util.*;

public class Evaluation {


  public static strictfp void main(String[] args) throws Exception {
    Configuration.setConfiguration(args.length > 0 ? args[0] : "JSONReader/config.properties");


    ArrayList<Integer> pS = new ArrayList<>(), pR = new ArrayList<>(), pD = new ArrayList<>();
    ArrayList<double[]> pxyz = new ArrayList<>();
    ArrayList<Integer> gS = new ArrayList<>(), gR = new ArrayList<>(), gD = new ArrayList<>();
    ArrayList<double[]> gxyz = new ArrayList<>();
    ArrayList<double[][]> final_worlds = new ArrayList<>();
    ArrayList<double[][]> start_worlds = new ArrayList<>();

    ArrayList<Task> data = LoadJSON.readJSON(Configuration.development);
    Double[][] Gold = Utils.readMatrix(TextFile.Read(Configuration.GoldData));
    readValues(Gold, gS, gR, gD, gxyz, false);

    switch (Configuration.baseline) {
      case None:
        Double[][] Test = Utils.readMatrix(TextFile.Read(Configuration.PredData));
        readValues(Test, pS, pR, pD, pxyz, true);
        break;
      case Center:
        pS.addAll(gS);
        double[] center = new double[]{0, 0.1, 0};
        for (int i = 0; i < gS.size(); ++i)
          pxyz.add(center);
        break;
      case Random:
        Random rand = new Random(20160329L);
        for (int i = 0; i < gS.size(); ++i) {
          pS.add(rand.nextInt(20));
          pR.add(rand.nextInt(20));
          pD.add(rand.nextInt(9));
          //pxyz.add(new double[] {rand.nextDouble()*2 - 1, rand.nextDouble()*2 - 1, rand.nextDouble()*2 - 1});
        }
        break;
      default:
        System.err.println("Invalid baseline: " + Configuration.baseline);
    }

    readWorlds(start_worlds, final_worlds, data);
    computePredictedXYZ(final_worlds, pR, pD, pxyz);
    computeGoldXYZ(final_worlds, gS, gxyz);
    incorporateSourcePredictionErrors(gS, pS, pxyz, start_worlds);

    evaluate(gS, pS, gR, pR, gD, pD, gxyz, pxyz);
  }

  public static void readValues(Double[][] Data, ArrayList<Integer> S, ArrayList<Integer> R, ArrayList<Integer> D,
                                ArrayList<double[]> xyz, boolean predicted) {
    int index = 0;
    for (Information information : Configuration.predict) {
      switch (information) {
        case Source:
          for (int i = 0; i < Data.length; ++i) {
            S.add(Data[i][index].intValue() - (predicted ? 1 : 0));
          }
          index += 1;
          break;
        case Reference:
          for (int i = 0; i < Data.length; ++i) {
            R.add(Data[i][index].intValue() - (predicted ? 1 : 0));
          }
          index += 1;
          break;
        case Direction:
          for (int i = 0; i < Data.length; ++i) {
            D.add(Data[i][index].intValue() - (predicted ? 1 : 0));
          }
          index += 1;
          break;
        case XYZ:
          for (int i = 0; i < Data.length; ++i) {
            xyz.add(new double[]{Data[i][index], Data[i][index + 1], Data[i][index + 2]});
          }
          index += 3;
        default:
          System.err.println("Invalid Prediction Type");
      }
    }
  }

  public static void readWorlds(ArrayList<double[][]> start_worlds, ArrayList<double[][]> final_worlds, ArrayList<Task> data) {
    // Collect worlds
    for (Task task : data) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String utterance : note.notes) {
            final_worlds.add(task.states[note.finish]);
            start_worlds.add(task.states[note.start]);
          }
        }
      }
    }
  }

  public static void computePredictedXYZ(ArrayList<double[][]> final_worlds,
                                         ArrayList<Integer> pR, ArrayList<Integer> pD, ArrayList<double[]> pxyz) {
    if (pxyz.isEmpty()) {
      double offset = 0.1666;
      double[] reference;
      double[] center = new double[]{0, 0.1, 0};
      for (int i = 0; i < final_worlds.size(); ++i) {
        if (pR.get(i) < final_worlds.get(i).length)
          reference = final_worlds.get(i)[pR.get(i)];
        else
          reference = center;
        switch (pD.get(i)) {
          case 6: // if dx < 0 and dz < 0 SW
            pxyz.add(new double[]{reference[0] - offset, reference[1], reference[2] - offset});
            break;
          case 3: // if dx < 0 and dz = 0 W
            pxyz.add(new double[]{reference[0] - offset, reference[1], reference[2]});
            break;
          case 0: // if dx < 0 and dz > 0 NW
            pxyz.add(new double[]{reference[0] - offset, reference[1], reference[2] + offset});
            break;
          case 7: // if dx = 0 and dz < 0 S
            pxyz.add(new double[]{reference[0], reference[1], reference[2] - offset});
            break;
          case 4: // if dx = 0 and dz = 0 TOP
            pxyz.add(new double[]{reference[0], reference[1], reference[2]});
            break;
          case 1: // if dx = 0 and dz > 0 N
            pxyz.add(new double[]{reference[0], reference[1], reference[2] + offset});
            break;
          case 8: // if dx > 0 and dz < 0 SE
            pxyz.add(new double[]{reference[0] + offset, reference[1], reference[2] - offset});
            break;
          case 5: // if dx > 0 and dz = 0 E
            pxyz.add(new double[]{reference[0] + offset, reference[1], reference[2]});
            break;
          case 2: // if dx > 0 and dz > 0 NE
            pxyz.add(new double[]{reference[0] + offset, reference[1], reference[2] + offset});
            break;
        }
      }
    }
  }

  public static void computeGoldXYZ(ArrayList<double[][]> final_worlds, ArrayList<Integer> gS, ArrayList<double[]> gxyz) {
    if (gxyz.isEmpty()) {
      for (int i = 0; i < final_worlds.size(); ++i) {
        gxyz.add(final_worlds.get(i)[gS.get(i)]);
      }
    }
  }

  public static void incorporateSourcePredictionErrors(ArrayList<Integer> gS, ArrayList<Integer> pS,
                                                       ArrayList<double[]> pxyz, ArrayList<double[][]> start_worlds) {
    // Update location information if source prediction is incorrect
    for (int i = 0; i < gS.size(); ++i) {
      if (gS.get(i) != pS.get(i))
        pxyz.set(i, start_worlds.get(i)[gS.get(i)]);
    }
  }

  public static void evaluate(ArrayList<Integer> gS, ArrayList<Integer> pS, ArrayList<Integer> gR,
                              ArrayList<Integer> pR, ArrayList<Integer> gD, ArrayList<Integer> pD,
                              ArrayList<double[]> gxyz, ArrayList<double[]> pxyz) {
    // Evaluate
    int eS = 0, eR = 0, eD = 0;
    ArrayList<Double> errors = new ArrayList<>();
    for (int i = 0; i < gS.size(); ++i) {
      eS += gS.get(i) == pS.get(i) ? 1 : 0;
      if (!pR.isEmpty()) {
        eR += gR.get(i) == pR.get(i) ? 1 : 0;
        eD += gD.get(i) == pD.get(i) ? 1 : 0;
      }
      errors.add(Utils.distance(pxyz.get(i), gxyz.get(i)));
    }
    System.out.println(String.format("Source %5.3f", 100.0*eS/gS.size()));
    if (!gR.isEmpty()) {
      System.out.println(String.format("Reference %5.3f", 100.0 * eR / gS.size()));
      System.out.println(String.format("Direction %5.3f", 100.0 * eD / gS.size()));
    }
    System.out.println(String.format("Mean Error: %5.3f", errors.stream().mapToDouble(j -> j).sum() / gS.size()));
    Collections.sort(errors);
    System.out.println(String.format("Median Error: %5.3f", errors.get(errors.size()/2)));

  }
}
