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
import java.util.List;
import java.util.stream.Collectors;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.FastBasicKeypointMatcher;
import org.openimaj.feature.local.matcher.LocalFeatureMatcher;
import org.openimaj.feature.local.matcher.MatchingUtilities;
import org.openimaj.feature.local.matcher.consistent.ConsistentLocalFeatureMatcher2d;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.local.engine.asift.ASIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.math.geometry.transforms.HomographyRefinement;
import org.openimaj.math.geometry.transforms.estimation.RobustHomographyEstimator;
import org.openimaj.math.model.fit.RANSAC;
import org.openimaj.util.pair.Pair;

public class ImageDiffDetector {

    public static void main(String[] args) throws IOException {

        final ASIFTEngine engine = new ASIFTEngine(false, 9);

        String dirLocation = "/opt/chewer/images/HDCVI_ch14_main_20210324090000_20210324090500.dav/";

        File first = new File(dirLocation + "10Faces9bfd77ab-38fc-4bfe-908b-4b324da7d8530.8500179161374743.jpg");
        FImage input = ImageUtilities.readF(first);
        LocalFeatureList<Keypoint> lfirst = engine.findKeypoints(input);
        LocalFeatureMatcher<Keypoint> matcher = createFastBasicMatcher();
        matcher.setModelFeatures(lfirst);

        List<File> files = Files.list(Paths.get(dirLocation))
                .map(Path::toFile)
                .collect(Collectors.toList());

        for (File file : files) {
            FImage ipt = ImageUtilities.readF(file);
            LocalFeatureList<Keypoint> lf = engine.findKeypoints(ipt);
            System.out.println(lfirst.size() + " -> " + lf.size() + " " + " NMatches: " + matcher.getMatches().size() + " " + matcher.findMatches(lf) + " " + file.getName());
        }

        // Prepare the matcher, uncomment this line to use a basic matcher as
        // opposed to one that enforces homographic consistency
        //  final LocalFeatureMatcher<Keypoint> matcher = createConsistentRANSACHomographyMatcher();
        // Find features in image 1
        // ... against image 2
//        matcher.findMatches(input2Feats);
        // Get the matches
//        final List<Pair<Keypoint>> matches = matcher.getMatches();
//        System.out.println("NMatches: " + matches.size());
        // Display the results
//        final MBFImage inp1MBF = input_1.toRGB();
//        final MBFImage inp2MBF = input_5.toRGB();
//        DisplayUtilities.display(MatchingUtilities.drawMatches(inp1MBF, inp2MBF, matches, RGBColour.BLUE));
    }

    /**
     * @return a matcher with a homographic constraint
     */
    private static LocalFeatureMatcher<Keypoint> createConsistentRANSACHomographyMatcher() {
        final ConsistentLocalFeatureMatcher2d<Keypoint> matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(
                createFastBasicMatcher());
        matcher.setFittingModel(new RobustHomographyEstimator(10.0, 1000, new RANSAC.BestFitStoppingCondition(),
                HomographyRefinement.NONE));

        return matcher;
    }

    /**
     * @return a basic matcher
     */
    private static LocalFeatureMatcher<Keypoint> createFastBasicMatcher() {
        return new FastBasicKeypointMatcher<Keypoint>(8);
    }
}
