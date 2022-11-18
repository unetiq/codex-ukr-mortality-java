# Package code to a jar file and execute the main class
mvn clean install
java -cp target/mortality-ukr-1.0-SNAPSHOT.jar de.unetiq.RiskScoreICU
