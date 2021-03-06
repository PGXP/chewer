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
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import org.opencv.core.Core;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class FaceDetectorVideo {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        System.out.println("\nRunning FaceDetector");

        Set<CascadeClassifier> classifierFaces = new HashSet<>();
        Set<CascadeClassifier> classifierBodys = new HashSet<>();

<<<<<<< HEAD
        String dirLocation = "/media/gladson/Elements/videos/14/";
        String dirFinal = "/media/gladson/Elements/vistos/";
        String dirOut = "/media/gladson/Elements/imagens/";
=======
        String dirLocation = "/media/desktop/Elements/videos/";
        String dirFinal = "/media/desktop/Elements/vistos/";
        String dirOut = "/media/desktop/Elements/imagens/novo/";
>>>>>>> 9b4647b3052f62ea7776a35b2619b9e9e6fc1b91
//        String dirLocation = "/opt/chewer/video/";
//        String dirFinal = "/opt/chewer/final/";
//        String dirOut = "/opt/chewer/images/";

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
                int i = 0;
                int p = 0;
                double avg = 1;
                double valor = 1;
                Path path = Paths.get(dirOut + file.getName());
                Files.createDirectories(path);
                Mat webcamFirst = null;

                if (capture.isOpened()) {
                    while (true) {

                        Mat webcamMatImage = new Mat();
                        capture.read(webcamMatImage);
                        if (!webcamMatImage.empty()) {
                            i++;
                            p++;

                            if (webcamFirst == null) {
                                webcamFirst = webcamMatImage;
                            }

                            if (i == 10) {
                                i = 0;

//                                avg = (valor + avg) / 2;
                                valor = compare(webcamFirst, webcamMatImage);

                                webcamFirst = webcamMatImage;
                                String encontrou = "";

                                if (valor <= 0.98) {

                                    avg = (valor + avg) / 2;
                                    List<Rect> rectsFace = new ArrayList<>();

                                    for (CascadeClassifier classifierFace : classifierFaces) {
                                        MatOfRect faceDetections = new MatOfRect();
                                        classifierFace.detectMultiScale(webcamMatImage, faceDetections, 1.3, 9);
                                        rectsFace.addAll(Arrays.asList(faceDetections.toArray()));
                                    }

                                    if (!rectsFace.isEmpty()) {
                                        encontrou = " Faces ";
                                    }

                                    for (Rect rect : rectsFace) {
                                        Imgproc.rectangle(webcamMatImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                                                new Scalar(255, 0, 0));
                                    }

                                    List<Rect> rectsBody = new ArrayList<>();

                                    for (CascadeClassifier classifierBody : classifierBodys) {
                                        MatOfRect faceDetections = new MatOfRect();
                                        classifierBody.detectMultiScale(webcamMatImage, faceDetections, 1.3, 9);
                                        rectsBody.addAll(Arrays.asList(faceDetections.toArray()));
                                    }

                                    if (!rectsBody.isEmpty()) {
                                        encontrou = encontrou + " Bodies ";
                                    }

                                    for (Rect rect : rectsBody) {
                                        Imgproc.rectangle(webcamMatImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                                                new Scalar(0, 255, 0));
                                    }

                                    Imgcodecs.imwrite(dirOut + file.getName() + "/" + p + encontrou + "  " + valor + " " + ".jpg", webcamMatImage);

                                }
//                               

                            }
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

    public static double compare(Mat oriBase, Mat oriTest1) {

        Mat hsvBase = new Mat(), hsvTest1 = new Mat();
        Mat srcBase = new Mat(), srcTest1 = new Mat();

        Imgproc.medianBlur(oriBase, srcBase, 7);
        Imgproc.medianBlur(oriTest1, srcTest1, 7);
        Imgproc.cvtColor(srcBase, hsvBase, Imgproc.COLOR_BGR2HSV);
        Imgproc.cvtColor(srcTest1, hsvTest1, Imgproc.COLOR_BGR2HSV);

        int hBins = 33, sBins = 33;
        int[] histSize = {hBins, sBins};
        float[] ranges = {0, 180, 0, 256};
        int[] channels = {0, 1};
        Mat histBase = new Mat(), histTest1 = new Mat();

        List<Mat> hsvBaseList = Arrays.asList(hsvBase);
        Imgproc.calcHist(hsvBaseList, new MatOfInt(channels), new Mat(), histBase, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histBase, histBase, 0, 1, Core.NORM_MINMAX);

        List<Mat> hsvTest1List = Arrays.asList(hsvTest1);
        Imgproc.calcHist(hsvTest1List, new MatOfInt(channels), new Mat(), histTest1, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histTest1, histTest1, 0, 1, Core.NORM_MINMAX);

        return Imgproc.compareHist(histBase, histTest1, 0);

    }
}
