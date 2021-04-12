/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br.com.pgxp;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Range;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class VideoDetector {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        System.out.println("\nRunning FaceDetector");

        Set<CascadeClassifier> classifierFaces = new HashSet<>();
        Set<CascadeClassifier> classifierBodys = new HashSet<>();

        String dirLocation = "/media/gladson/Elements/videos/16/";
        String dirFinal = "/media/gladson/Elements/vistos/";
        String dirOut = "/media/gladson/Elements/imagens/";
        String dirClassifierFace = "/opt/chewer/face/";
        String dirClassifierBody = "/opt/chewer/body/";

        try {
            List<File> files = Files.list(Paths.get(dirLocation))
                    .map(Path::toFile)
                    .collect(Collectors.toList());

            List<File> faces = Files.list(Paths.get(dirClassifierFace))
                    .map(Path::toFile)
                    .collect(Collectors.toList());

            List<File> bodys = Files.list(Paths.get(dirClassifierBody))
                    .map(Path::toFile)
                    .collect(Collectors.toList());

            for (File face : faces) {
                classifierFaces.add(new CascadeClassifier(dirClassifierFace + face.getName()));
            }

            for (File face : bodys) {
                classifierBodys.add(new CascadeClassifier(dirClassifierBody + face.getName()));
            }

            for (File file : files) {

                VideoCapture capture = new VideoCapture(dirLocation + file.getName());

                Path path = Paths.get(dirOut + file.getName());
                Files.createDirectories(path);

                List<Mat> imagens = new ArrayList<>();

                if (capture.isOpened()) {
                    while (true) {
                        Mat webcamMatImage = new Mat();

                        capture.read(webcamMatImage);

                        if (!webcamMatImage.empty()) {
                            imagens.add(webcamMatImage);
                        } else {
                            System.out.println(" -- Frame not captured -- Break!" + file.getName());
                            capture.release();
                            Files.move(Paths.get(dirLocation + file.getName()), Paths.get(dirFinal + file.getName()), StandardCopyOption.REPLACE_EXISTING);
                            System.gc();
                            break;
                        }

                    }
                } else {
                    System.out.println("Couldn't open capture.");
                }

                for (int i = 1; i < imagens.size() - 1; i++) {
                    compare(imagens.get(0), imagens.get(i), imagens.get(i+1));
                }

            }

        } catch (IOException e) {
            // Error while reading the directory
        }
        System.out.println("Image Processed");
    }

    private static String createDirectory(String foler, String filename) {

        String newfolder = filename.replace(".dat", "");

        try {

            Path path = Paths.get(foler + newfolder);
            Files.createDirectories(path);
            path = Paths.get(foler + newfolder + "/bodies");
            Files.createDirectories(path);
            path = Paths.get(foler + newfolder + "/faces");
            Files.createDirectories(path);

        } catch (IOException e) {
            System.err.println("Failed to create directory!" + e.getMessage());
        }

        return foler + newfolder;

    }

    public static void compare(Mat srcBase, Mat srcTest1, Mat srcTest2) {

        if (srcBase.empty() || srcTest1.empty() || srcTest2.empty()) {
            System.err.println("Cannot read the images");
        }

        Mat hsvBase = new Mat(), hsvTest1 = new Mat(), hsvTest2 = new Mat();
        Imgproc.cvtColor(srcBase, hsvBase, Imgproc.COLOR_BGR2HSV);
        Imgproc.cvtColor(srcTest1, hsvTest1, Imgproc.COLOR_BGR2HSV);
        Imgproc.cvtColor(srcTest2, hsvTest2, Imgproc.COLOR_BGR2HSV);
        Mat hsvHalfDown = hsvBase.submat(new Range(hsvBase.rows() / 2, hsvBase.rows() - 1), new Range(0, hsvBase.cols() - 1));
        int hBins = 50, sBins = 60;
        int[] histSize = {hBins, sBins};
        // hue varies from 0 to 179, saturation from 0 to 255
        float[] ranges = {0, 180, 0, 256};
        // Use the 0-th and 1-st channels
        int[] channels = {0, 1};
        Mat histBase = new Mat(), histHalfDown = new Mat(), histTest1 = new Mat(), histTest2 = new Mat();
        List<Mat> hsvBaseList = Arrays.asList(hsvBase);
        Imgproc.calcHist(hsvBaseList, new MatOfInt(channels), new Mat(), histBase, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histBase, histBase, 0, 1, Core.NORM_MINMAX);
        List<Mat> hsvHalfDownList = Arrays.asList(hsvHalfDown);
        Imgproc.calcHist(hsvHalfDownList, new MatOfInt(channels), new Mat(), histHalfDown, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histHalfDown, histHalfDown, 0, 1, Core.NORM_MINMAX);
        List<Mat> hsvTest1List = Arrays.asList(hsvTest1);
        Imgproc.calcHist(hsvTest1List, new MatOfInt(channels), new Mat(), histTest1, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histTest1, histTest1, 0, 1, Core.NORM_MINMAX);
        List<Mat> hsvTest2List = Arrays.asList(hsvTest2);
        Imgproc.calcHist(hsvTest2List, new MatOfInt(channels), new Mat(), histTest2, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histTest2, histTest2, 0, 1, Core.NORM_MINMAX);
        for (int compareMethod = 0; compareMethod < 4; compareMethod++) {
            double baseTest1 = Imgproc.compareHist(histBase, histTest1, compareMethod);
            double baseTest2 = Imgproc.compareHist(histTest1, histTest2, compareMethod);
            System.out.println(" / " + baseTest1 + " / " + baseTest2+ " / " +(baseTest2-baseTest1));
        }
    }

}
