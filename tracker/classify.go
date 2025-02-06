// Package tracker implements an object tracker as a Viam vision service
// This file contains methods that are useful for classifying the detections
package tracker

import (
	"context"
	"image"
	"image/draw"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/services/vision"
	objdet "go.viam.com/rdk/vision/objectdetection"
)

// method will take a slice of tracks and return a slice of tracks.
// The only difference is that the track will now include the detection classification
func classifyTracks(ctx context.Context, tracks []*track, img image.Image, classifier vision.Service, logger logging.Logger) []*track {

	if classifier == nil {
		return tracks
	}

	for _, tr := range tracks {
		cropped := cropImageFromDet(img, tr.Det)
		out, err := classifier.Classifications(ctx, cropped, 1, nil)
		if err != nil || len(out) < 1 {
			// if there is an error, just skip the classification
			logger.Warnf("error classifying detection: %v", err)
			continue
		}
		tr.detClassification = out[0]
	}
	return tracks
}

// empty bounding box implies no crop
func cropImageFromDet(img image.Image, det objdet.Detection) image.Image {
	bb := det.BoundingBox()
	if bb.Max.X == 0 || bb.Max.Y == 0 {
		return img
	}
	// Get image bounds
	bounds := img.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y

	// Convert relative coordinates to absolute pixels
	x1 := bounds.Min.X + int(float64(bb.Min.X)*float64(width))
	y1 := bounds.Min.Y + int(float64(bb.Min.Y)*float64(height))
	x2 := bounds.Min.X + int(float64(bb.Max.X)*float64(width))
	y2 := bounds.Min.Y + int(float64(bb.Max.X)*float64(height))

	// Create cropping rectangle
	rect := image.Rect(x1, y1, x2, y2)
	croppedImg := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	draw.Draw(croppedImg, croppedImg.Bounds(), img, rect.Min, draw.Src)
	return croppedImg
}

/*
func cropImageFromDet2(img image.Image, det objdet.Detection) image.Image {
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
*/
