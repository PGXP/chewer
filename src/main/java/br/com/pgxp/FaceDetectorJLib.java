/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br.com.pgxp;

import com.emaraic.jdlib.Jdlib;
import com.emaraic.utils.FaceDescriptor;
import com.emaraic.utils.ImageUtils;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;

public class FaceDetectorJLib {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        try {
            String dirLocation = "/home/desktop/Imagens/web/";
            String dirOut = "/home/desktop/Imagens/out/";

            List<File> files = Files.list(Paths.get(dirLocation))
                    .map(Path::toFile)
                    .collect(Collectors.toList());

            String facialLandmarksModelPath = "/opt/chewer/dlib/shape_predictor_68_face_landmarks.dat";
            Jdlib jdlib = new Jdlib(facialLandmarksModelPath);

            for (File file : files) {

                BufferedImage img = loadImage(dirLocation + file.getName());

                ArrayList<FaceDescriptor> faces = jdlib.getFaceLandmarks(img);

                for (FaceDescriptor face : faces) {
                    ImageUtils.drawFaceDescriptor(img, face);
                    ImageIO.write(img, "jpg", new File(dirOut + "_new_" + file.getName()));
                }

//                img = resize(img, 800, 800);
//                ImageIO.write(img, "jpg", file);
//                Imgcodecs.imwrite(dirOut + file.getName() + "_out.jpg", img);
            }
        } catch (IOException ex) {
            Logger.getLogger(FaceDetectorJLib.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    private static BufferedImage loadImage(String imagepath) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(imagepath));
        } catch (IOException e) {
            System.err.println("Error During Loading File: " + imagepath);
        }
        return img;
    }

    public static BufferedImage resize(BufferedImage img, int w, int h) {
        Image tempimg = img.getScaledInstance(w, h, Image.SCALE_SMOOTH);
        BufferedImage image = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = image.createGraphics();
        g2d.drawImage(tempimg, 0, 0, null);
        g2d.dispose();
        return image;
    }

}
