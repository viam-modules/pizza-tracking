package tracker

import (
	"strconv"
	"strings"

	"github.com/pkg/errors"
	objdet "go.viam.com/rdk/vision/objectdetection"
)

type track struct {
	Det              objdet.Detection
	persistenceLimit int
	persistenceCount int
	stable           bool
}

func newTrack(det objdet.Detection, lim int) *track {
	return &track{det, lim, 0, false}
}

func newTracks(dets []objdet.Detection, lim int) []*track {
	tracks := make([]*track, 0, len(dets))
	for _, d := range dets {
		tracks = append(tracks, newTrack(d, lim))
	}
	return tracks
}

func (tr *track) clone() *track {
	return &track{
		tr.Det,
		tr.persistenceLimit,
		tr.persistenceCount,
		tr.stable,
	}
}

func (tr *track) isStable() bool {
	return tr.stable
}

func (tr *track) addPersistence() {
	if tr.stable {
		return
	}
	tr.persistenceCount += 1
	if tr.persistenceCount >= tr.persistenceLimit {
		tr.stable = true
	}
}

func getStableDetections(tracks []*track) []objdet.Detection {
	dets := make([]objdet.Detection, 0, len(tracks))
	for _, tr := range tracks {
		if tr.stable {
			dets = append(dets, tr.Det)
		}
	}
	return dets
}

func getAllDetections(tracks []*track) []objdet.Detection {
	dets := make([]objdet.Detection, 0, len(tracks))
	for _, tr := range tracks {
		dets = append(dets, tr.Det)
	}
	return dets
}

type trackedObject struct {
	FullLabel string
	Label     string
	Id        int
	Time      string
}

func newTrackedObjectFromLabel(label string) (trackedObject, error) {
	parts := strings.Split(label, "_")
	id, err := strconv.Atoi(parts[1])
	if err != nil {
		return trackedObject{}, errors.Wrapf(err, "unable to parse label %v", label)
	}
	return trackedObject{
		FullLabel: label,
		Label:     parts[0],
		Id:        id,
		Time:      strings.Join(parts[2:], "_"),
	}, nil

}
