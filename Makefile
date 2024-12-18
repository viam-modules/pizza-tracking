
module.tar.gz:
	go build -a -o module main.go
	tar -czf $@ module
