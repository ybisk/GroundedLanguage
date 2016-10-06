import JSON.Note;
import JSON.Task;

import java.util.*;

public class Evaluation {


  public static strictfp void main(String[] args) throws Exception {
    Configuration.setConfiguration(args.length > 0 ? args[0] : "config.properties");

    ArrayList<Integer> pS = new ArrayList<>(), pR = new ArrayList<>(), pD = new ArrayList<>();
    ArrayList<double[]> pTxyz = new ArrayList<>();
    ArrayList<double[]> pSxyz = new ArrayList<>();
    ArrayList<Integer> gS = new ArrayList<>(), gR = new ArrayList<>(), gD = new ArrayList<>();
    ArrayList<double[]> gTxyz = new ArrayList<>();
    ArrayList<double[]> gSxyz = new ArrayList<>();
    ArrayList<String> utterances = new ArrayList<>();
    ArrayList<String> images = new ArrayList<>();
    ArrayList<double[][]> final_worlds = new ArrayList<>();
    ArrayList<double[][]> start_worlds = new ArrayList<>();

    ArrayList<Task> data = LoadJSON.readJSON(Configuration.testing);
    Double[][] Gold = Utils.readMatrix(TextFile.Read(Configuration.GoldData));
    readValues(Gold, gS, gR, gD, gTxyz, gSxyz, false);

    switch (Configuration.baseline) {
      case None:
        Double[][] Test = Utils.readMatrix(TextFile.Read(Configuration.PredData));
        System.out.println(Configuration.PredData);
        readValues(Test, pS, pR, pD, pTxyz, pSxyz, true);
        break;
      case Center:
        pS.addAll(gS);
        double[] center = new double[]{0, 0.1, 0};
        for (int i = 0; i < gS.size(); ++i)
          pTxyz.add(center);
        break;
      case Random:
        Random rand = new Random(20160329L);
        for (int i = 0; i < gS.size(); ++i) {
          pS.add(rand.nextInt(Configuration.blocktype.equals(Configuration.BlockType.Random) ? 10 : 20));
          pR.add(rand.nextInt(Configuration.blocktype.equals(Configuration.BlockType.Random) ? 10 : 20));
          pD.add(rand.nextInt(9));
        }
        break;
      case Oracle:
        pS.addAll(gS);
        pR.addAll(gR);
        pD.addAll(gD);
        break;
      default:
        System.err.println("Invalid baseline: " + Configuration.baseline);
    }

    readTaskData(start_worlds, final_worlds, utterances, images, data);
    computePredictedTargetXYZ(final_worlds, pR, pD, pTxyz);
    computePredictedSourceXYZ(start_worlds, pS, pSxyz);
    computeGoldXYZ(final_worlds, gS, gTxyz);
    computeGoldXYZ(start_worlds, gS, gSxyz);
    incorporateSourcePredictionErrors(gS, pS, pTxyz, start_worlds);

    ArrayList<Tuple<Integer>> errors = evaluate(gS, pS, gR, pR, gD, pD, gTxyz, pTxyz, gSxyz, pSxyz);

    worstOffenders(errors, images, gS, pS, gR, pR, gD, pD, gTxyz, pTxyz, utterances);
  }

  public static void readValues(Double[][] Data, ArrayList<Integer> S, ArrayList<Integer> R, ArrayList<Integer> D,
                                ArrayList<double[]> Txyz, ArrayList<double[]> Sxyz, boolean predicted) {
    int index = 0;
    for (Information information : Configuration.predict) {
      switch (information) {
        case Source:
          for (int i = 0; i < Data.length; ++i) {
            S.add(Data[i][index].intValue() - (!predicted ? 1 : 0));
          }
          index += 1;
          break;
        case Reference:
          for (int i = 0; i < Data.length; ++i) {
            R.add(Data[i][index].intValue() - (!predicted ? 1 : 0));
          }
          index += 1;
          break;
        case Direction:
          for (int i = 0; i < Data.length; ++i) {
            D.add(Data[i][index].intValue() - (!predicted ? 1 : 0));
          }
          index += 1;
          break;
        case sXYZ:
          for (int i = 0; i < Data.length; ++i) {
            Sxyz.add(new double[]{Data[i][index], Data[i][index + 1], Data[i][index + 2]});
          }
          index += 3;
          break;
        case tXYZ:
          for (int i = 0; i < Data.length; ++i) {
            Txyz.add(new double[]{Data[i][index], Data[i][index + 1], Data[i][index + 2]});
          }
          index += 3;
          break;
        default:
          System.err.println("Invalid Prediction Type");
      }
    }
  }

  public static void readTaskData(ArrayList<double[][]> start_worlds, ArrayList<double[][]> final_worlds,
                                  ArrayList<String> utterances, ArrayList<String> images, ArrayList<Task> data) {
    // Collect worlds
    for (Task task : data) {
      for (Note note : task.notes) {
        if (note.type.equals("A0")) {
          for (String utterance : note.notes) {
            final_worlds.add(task.states[note.finish]);
            start_worlds.add(task.states[note.start]);
            utterances.add(utterance);
            images.add(task.images[note.start] + " - " + task.images[note.finish]);
          }
        }
      }
    }
  }

  public static void computePredictedSourceXYZ(ArrayList<double[][]> final_worlds,
                                               ArrayList<Integer> pS, ArrayList<double[]> pSxyz) {
    if (!pS.isEmpty()) {
      double[] center = new double[]{0, 0.1, 0};
      for (int i = 0; i < final_worlds.size(); ++i) {
        if (pS.get(i) < final_worlds.get(i).length)
          pSxyz.add(final_worlds.get(i)[pS.get(i)]);
        else
          pSxyz.add(center);
      }
    }
  }
  public static void computePredictedTargetXYZ(ArrayList<double[][]> final_worlds,
                                              ArrayList<Integer> pR, ArrayList<Integer> pD, ArrayList<double[]> pTxyz) {
    if (pTxyz.isEmpty()) {
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
            pTxyz.add(new double[]{reference[0] - offset, reference[1], reference[2] - offset});
            break;
          case 3: // if dx < 0 and dz = 0 W
            pTxyz.add(new double[]{reference[0] - offset, reference[1], reference[2]});
            break;
          case 0: // if dx < 0 and dz > 0 NW
            pTxyz.add(new double[]{reference[0] - offset, reference[1], reference[2] + offset});
            break;
          case 7: // if dx = 0 and dz < 0 S
            pTxyz.add(new double[]{reference[0], reference[1], reference[2] - offset});
            break;
          case 4: // if dx = 0 and dz = 0 TOP
            pTxyz.add(new double[]{reference[0], reference[1] + offset, reference[2]});
            break;
          case 1: // if dx = 0 and dz > 0 N
            pTxyz.add(new double[]{reference[0], reference[1], reference[2] + offset});
            break;
          case 8: // if dx > 0 and dz < 0 SE
            pTxyz.add(new double[]{reference[0] + offset, reference[1], reference[2] - offset});
            break;
          case 5: // if dx > 0 and dz = 0 E
            pTxyz.add(new double[]{reference[0] + offset, reference[1], reference[2]});
            break;
          case 2: // if dx > 0 and dz > 0 NE
            pTxyz.add(new double[]{reference[0] + offset, reference[1], reference[2] + offset});
            break;
        }
      }
    }
  }

  public static void computeGoldXYZ(ArrayList<double[][]> final_worlds, ArrayList<Integer> gS, ArrayList<double[]> gxyz) {
    if (gxyz.isEmpty()) {
      for (int i = 0; i < final_worlds.size(); ++i) {
        System.out.println(i + "\t" + gS.get(i));
        System.out.println(final_worlds.get(i).length);
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

  public static ArrayList<Tuple<Integer>> evaluate(ArrayList<Integer> gS, ArrayList<Integer> pS, ArrayList<Integer> gR,
                              ArrayList<Integer> pR, ArrayList<Integer> gD, ArrayList<Integer> pD,
                              ArrayList<double[]> gTxyz, ArrayList<double[]> pTxyz,
                              ArrayList<double[]> gSxyz, ArrayList<double[]> pSxyz) {
    // Evaluate
    int eS = 0, eR = 0, eD = 0;
    ArrayList<Double> errors = new ArrayList<>();
    ArrayList<Double> Serrors = new ArrayList<>();
    ArrayList<Tuple<Integer>> errorIDs = new ArrayList<>();
    for (int i = 0; i < gSxyz.size(); ++i) {
      if (!pR.isEmpty()) {
        eS += gS.get(i) == pS.get(i) ? 1 : 0;
        eR += gR.get(i) == pR.get(i) ? 1 : 0;
        eD += gD.get(i) == pD.get(i) ? 1 : 0;
      }
      errors.add(Utils.distance(pTxyz.get(i), gTxyz.get(i)));
      Serrors.add(Utils.distance(pSxyz.get(i), gSxyz.get(i)));
      errorIDs.add(new Tuple<>(i,Utils.distance(pTxyz.get(i), gTxyz.get(i))));
    }
    System.out.println(String.format("Source %5.3f", 100.0*eS/gS.size()));
    if (!gR.isEmpty()) {
      System.out.println(String.format("Reference %5.3f", 100.0 * eR / gS.size()));
      System.out.println(String.format("Direction %5.3f", 100.0 * eD / gS.size()));
    }
    System.out.println("Source");
    System.out.println(String.format("Mean Error: %5.3f", Serrors.stream().mapToDouble(j -> j).sum() / Serrors.size()));
    Collections.sort(Serrors);
    System.out.println(String.format("Median Error: %5.3f", Serrors.get(Serrors.size()/2)));
    System.out.println("Target");
    System.out.println(String.format("Mean Error: %5.3f", errors.stream().mapToDouble(j -> j).sum() / errors.size()));
    Collections.sort(errors);
    System.out.println(String.format("Median Error: %5.3f", errors.get(errors.size()/2)));
    return errorIDs;
  }

  private static void worstOffenders(ArrayList<Tuple<Integer>> errors, ArrayList<String> images,
                                     ArrayList<Integer> gS, ArrayList<Integer> pS,
                                     ArrayList<Integer> gR, ArrayList<Integer> pR,
                                     ArrayList<Integer> gD, ArrayList<Integer> pD,
                                     ArrayList<double[]> gxyz, ArrayList<double[]> pxyz, ArrayList<String> utterances) {
    // Sort Errors
    Collections.sort(errors);
    Collections.reverse(errors);

    System.out.println();
    int idx;
    double err;
    for (int i = 0; i < 55; ++i) {
      err = errors.get(i).value();
      idx = errors.get(i).content();
      if (gR.isEmpty())
        System.out.println(String.format("%8.5f %s %s", err, images.get(idx), utterances.get(idx)));
      else
        System.out.println(String.format("%8.5f %s %5b %5b %5b %-14s %-14s %-2s %s", err,
            images.get(idx), gS.get(idx) == pS.get(idx), gR.get(idx) == pR.get(idx),
            gD.get(idx) == pD.get(idx), CreateTrainingData.brands.get(gS.get(idx)), CreateTrainingData.brands.get(gR.get(idx)),
            CreateTrainingData.cardinal[gD.get(idx)],  utterances.get(idx)));
    }
  }

}
