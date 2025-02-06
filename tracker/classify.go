// Package tracker implements an object tracker as a Viam vision service
// This file contains methods that are useful for classifying the detections
package tracker

import (
	"context"
	"image"

	"go.viam.com/rdk/rimage"
	"go.viam.com/rdk/services/vision"
	objdet "go.viam.com/rdk/vision/objectdetection"
)

// method will take a slice of tracks and return a slice of tracks.
// The only difference is that the track will now include the detection classification
func classifyTracks(ctx context.Context, tracks []*track, img image.Image, classifier vision.Service) []*track {
	
	for _, tr := range tracks {
		cropped := cropImageFromDet(img, tr.Det)
		out, err := classifier.Classifications(ctx, cropped, 1, nil)
		if err != nil || len(out) > 1 {
			// if there is an error, just skip the classification
			continue
		}
		tr.detClassification = out[0]
	}

	return tracks
}

func cropImageFromDet(img image.Image, det objdet.Detection) image.Image {
	lazyImg, ok := img.(*rimage.LazyEncodedImage)
	if ok {
		im := lazyImg.DecodedImage()
		cropped := im.(interface {
			SubImage(r image.Rectangle) image.Image
		}).SubImage(*det.BoundingBox())
		return cropped
	}

	return nil
}
