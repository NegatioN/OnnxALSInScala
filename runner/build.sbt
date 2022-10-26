ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

lazy val root = (project in file("."))
  .settings(
    name := "runner"
  )

libraryDependencies += "com.microsoft.onnxruntime" % "onnxruntime" % "1.12.1"
