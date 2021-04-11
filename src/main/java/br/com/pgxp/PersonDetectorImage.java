/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br.com.pgxp;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

public class PersonDetectorImage {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static List<String> getOutputNames(Net net) {

        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));//unfold and create R-CNN layers from the loaded YOLO model//
        return names;
    }

    public static void main(String[] args) throws InterruptedException {

        try {
            String modelWeights = "/opt/chewer/yolo/yolov3.weights"; //Download and load only wights for YOLO , this is obtained from official YOLO site//
            String modelConfiguration = "/opt/chewer/yolo/yolov3.cfg";//Download and load cfg file for YOLO , can be obtained from official site//

            Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);

            String dirLocation = "/home/desktop/Imagens/ipcam/";
            String dirOut = "/home/desktop/Imagens/out/";
            String dirClassifierFace = "/opt/chewer/face/";
            String dirClassifierBody = "/opt/chewer/body/";

            List<File> files = Files.list(Paths.get(dirLocation))
                    .map(Path::toFile)
                    .collect(Collectors.toList());

            for (File file : files) {

                Mat image = Imgcodecs.imread(dirLocation + file.getName());
                Size sz = new Size(288, 288);

                List<Mat> result = new ArrayList<>();
                List<String> outBlobNames = getOutputNames(net);

                Mat blob = Dnn.blobFromImage(image, 0.00392, sz, new Scalar(0), true, false); // We feed one frame of video into the network at a time, we have to convert the image to a blob. A blob is a pre-processed image that serves as the input.//
                net.setInput(blob);

                net.forward(result, outBlobNames); //Feed forward the model to get output //

                float confThreshold = 0.6f; //Insert thresholding beyond which the model will detect objects//
                List<Integer> clsIds = new ArrayList<>();
                List<Float> confs = new ArrayList<>();
                List<Rect> rects = new ArrayList<>();
                for (int i = 0; i < result.size(); ++i) {

                    Mat level = result.get(i);
                    for (int j = 0; j < level.rows(); ++j) {
                        Mat row = level.row(j);
                        Mat scores = row.colRange(5, level.cols());
                        Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                        float confidence = (float) mm.maxVal;
                        Point classIdPoint = mm.maxLoc;
                        if (confidence > confThreshold) {
//                            int centerX = (int) (row.get(0, 0)[0] * frame.cols()); //scaling for drawing the bounding boxes//
//                            int centerY = (int) (row.get(0, 1)[0] * frame.rows());
//                            int width = (int) (row.get(0, 2)[0] * frame.cols());
//                            int height = (int) (row.get(0, 3)[0] * frame.rows());
//                            int left = centerX - width / 2;
//                            int top = centerY - height / 2;

                            clsIds.add((int) classIdPoint.x);
                            confs.add((float) confidence);
//                            rects.add(new Rect(left, top, width, height));
                        }
                    }
                }
                float nmsThresh = 0.5f;
                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
                Rect[] boxesArray = rects.toArray(new Rect[0]);
                MatOfRect boxes = new MatOfRect(boxesArray);
                MatOfInt indices = new MatOfInt();
//                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); //We draw the bounding boxes for objects here//

                int[] ind = indices.toArray();
                int j = 0;
                for (int i = 0; i < ind.length; ++i) {
                    int idx = ind[i];
                    Rect box = boxesArray[idx];
                    Imgproc.rectangle(image, box.tl(), box.br(), new Scalar(0, 0, 255), 2);
                    //i=j;

                    System.out.println(idx);
                }
                // Imgcodecs.imwrite("D://out.png", image);
                //System.out.println("Image Loaded");
                //ImageIcon image = new ImageIcon(Mat2bufferedImage(frame)); //setting the results into a frame and initializing it //
                Imgcodecs.imwrite(dirOut + file.getName() + "_body.jpg", image);
            }

        } catch (IOException ex) {
            Logger.getLogger(PersonDetectorImage.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

//	}
    private static BufferedImage Mat2bufferedImage(Mat image) {   // The class described here  takes in matrix and renders the video to the frame  //
        MatOfByte bytemat = new MatOfByte();
        Imgcodecs.imencode(".jpg", image, bytemat);
        byte[] bytes = bytemat.toArray();
        InputStream in = new ByteArrayInputStream(bytes);
        BufferedImage img = null;
        try {
            img = ImageIO.read(in);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return img;
    }

//    static {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//    }
//
//    public static void main(String[] args) {
//
//        System.out.println("\nRunning FaceDetector");
//
//        Set<CascadeClassifier> classifierFaces = new HashSet<>();
//        Set<CascadeClassifier> classifierBodys = new HashSet<>();
//
//        String dirLocation = "/home/desktop/Imagens/ipcam/";
//        String dirOut = "/home/desktop/Imagens/out/";
//        String dirClassifierFace = "/opt/chewer/face/";
//        String dirClassifierBody = "/opt/chewer/body/";
//
//        try {
//
//            List<File> files = Files.list(Paths.get(dirLocation))
//                    .map(Path::toFile)
//                    .collect(Collectors.toList());
//
//            List<File> faces = Files.list(Paths.get(dirClassifierFace))
//                    .map(Path::toFile)
//                    .collect(Collectors.toList());
//
//            List<File> bodys = Files.list(Paths.get(dirClassifierBody))
//                    .map(Path::toFile)
//                    .collect(Collectors.toList());
//
//            for (File face : faces) {
//                classifierFaces.add(new CascadeClassifier(dirClassifierFace + face.getName()));
//            }
//
//            for (File face : bodys) {
//                classifierBodys.add(new CascadeClassifier(dirClassifierBody + face.getName()));
//            }
//
//            for (File file : files) {
//
//                List<Rect> rectsBody = new ArrayList<>();
//                List<Rect> rectsFace = new ArrayList<>();
//
//                Mat src = Imgcodecs.imread(dirLocation + file.getName());
//
//                for (CascadeClassifier classifierFace : classifierFaces) {
//                    MatOfRect faceDetections = new MatOfRect();
//                    classifierFace.detectMultiScale(src, faceDetections, 1.2, 9);
//                    rectsBody.addAll(Arrays.asList(faceDetections.toArray()));
//                }
//
//                if (!rectsBody.isEmpty()) {
//
//                    for (Rect rect : rectsBody) {
//                        Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
//                                new Scalar(255, 0, 0));
//                    }
//
//                    Imgcodecs.imwrite(dirOut + file.getName() + "_body.jpg", src);
//                }
//
//                for (CascadeClassifier classifierBody : classifierBodys) {
//                    MatOfRect faceDetections = new MatOfRect();
//                    classifierBody.detectMultiScale(src, faceDetections, 1.2, 9);
//                    rectsFace.addAll(Arrays.asList(faceDetections.toArray()));
//                }
//
//                if (!rectsFace.isEmpty()) {
//
//                    for (Rect rect : rectsFace) {
//                        Imgproc.rectangle(src, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
//                                new Scalar(255, 255, 0));
//                    }
//
//                    Imgcodecs.imwrite(dirOut + file.getName() + "_face.jpg", src);
//                }
//
//            }
//
//        } catch (IOException e) {
//            // Error while reading the directory
//        }
//
//        System.out.println("Image Processed");
//    }
}
