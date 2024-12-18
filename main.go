// Package main is a module which serves the mybase custom model.
package main

import (
	"context"
	"go.viam.com/rdk/services/vision"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/module"
	"go.viam.com/utils"

	"github.com/viam-modules/pizza-tracking/tracker"
)

func main() {
	module.ModularMain(resource.APIModel{API: vision.API, Model: tracker.Model})
}
