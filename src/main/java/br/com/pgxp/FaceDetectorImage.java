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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetectorImage {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        System.out.println("\nRunning FaceDetector");

        Set<CascadeClassifier> classifierFaces = new HashSet<>();
        Set<CascadeClassifier> classifierBodys = new HashSet<>();

        String dirLocation = "/home/desktop/Imagens/ipcam/";
        String dirOut = "/home/desktop/Imagens/out/";
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

                List<Rect> rectsBody = new ArrayList<>();
                List<Rect> rectsFace = new ArrayList<>();

                Mat src = Imgcodecs.imread(dirLocation + file.getName());

                for (CascadeClassifier classifierFace : classifierFaces) {
                    MatOfRect faceDetections = new MatOfRect();
                    classifierFace.detectMultiScale(src, faceDetections, 1.2, 9);
                    rectsBody.addAll(Arrays.asList(faceDetections.toArray()));
                }

                if (!rectsBody.isEmpty()) {

                    for (Rect rect : rectsBody) {
                        Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                                new Scalar(255, 0, 0));
                    }

                    Imgcodecs.imwrite(dirOut + file.getName() + "_body.jpg", src);
                }

                for (CascadeClassifier classifierBody : classifierBodys) {
                    MatOfRect faceDetections = new MatOfRect();
                    classifierBody.detectMultiScale(src, faceDetections, 1.2, 9);
                    rectsFace.addAll(Arrays.asList(faceDetections.toArray()));
                }

                if (!rectsFace.isEmpty()) {

                    for (Rect rect : rectsFace) {
                        Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                                new Scalar(255, 255, 0));
                    }

                    Imgcodecs.imwrite(dirOut + file.getName() + "_face.jpg", src);
                }

            }

        } catch (IOException e) {
            // Error while reading the directory
        }

        System.out.println("Image Processed");
    }
}
