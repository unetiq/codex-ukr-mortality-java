package de.unetiq;

import com.fasterxml.jackson.databind.ObjectMapper;

import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ListIterator;
import java.nio.file.Paths;

public class PatientDataHolder {

    /**
     * A vessel for processing data for multiple patients from a data input file.
     */

    public String model_features_path = "src/main/resources/data/model_features.json";
    public int urine_output_index = 20;
    public String urine_output_loinc = "29463-7";

    public Map<String, Map<String, Object>> model_features;
    private Map<String, List<Map<String, Object>>> patient_data;
    public int n_features;
    public int n_patients;

    public PatientDataHolder(String patient_data_path) throws java.io.IOException {

        /**
         * Read model features and the provided patient data file during initialization.
         *
         * @param patient_data_path Path to the patient data file.
         */

        this.model_features = read_model_features(model_features_path);
        this.n_features = this.model_features.size();
        this.patient_data = read_data_file(patient_data_path);
        this.n_patients = this.patient_data.size();

    }

    public float[] process_all_patients() throws java.io.IOException, XGBoostError {

        /**
         * Iterate through all patients and process their data.
         * This includes the normalization step for urine output (from ml to ml/kg).
         * 
         * @return Values for all patients in a flattened array.
         */

        float[] data = new float[this.n_patients * this.n_features];

        // Iterate through patients
        for (Map.Entry<String, List<Map<String, Object>>> patient_entry : this.patient_data.entrySet()) {

            // Parse index and available data points
            int patient_index = Integer.parseInt(patient_entry.getKey());
            List<Map<String, Object>> patient_data_points = (List<Map<String, Object>>) patient_entry.getValue();

            float[] patient_values = process_single_patient(patient_index, patient_data_points);

            // *** Urine output is obtained in ml, but the model expects it in normalized
            // form (ml/kg) ***
            float patient_weight = get_patient_weight(patient_data_points);
            patient_values[this.urine_output_index] = patient_values[this.urine_output_index]
                    / patient_weight;

            // Add values to patients data pool
            for (int i = 0; i < patient_values.length; i++) {
                data[patient_index * this.n_features + i] = patient_values[i];
            }

        }

        return data;
    }

    private float[] process_single_patient(int patient_index, List<Map<String, Object>> patient_data_points)
            throws XGBoostError, java.io.IOException {

        /**
         * Match patient data to required model features via LOINC codes and return an
         * array of the patient's data in the correct order.
         * 
         * @param patient_index       The index of the patient (important for retaining
         *                            order of the predictions).
         * @param patient_data_points A list of data points available for a single
         *                            patient as read from file.
         * @return An array of the patient's data in the order required by the model.
         */

        float[] patient_feature_values = new float[this.n_features];

        // Iterate through required features
        for (Map.Entry<String, Map<String, Object>> feature : this.model_features.entrySet()) {

            // Parse the current feature's LOINC code and aggregation method
            int feature_index = Integer.parseInt(feature.getKey());
            Map<String, Object> current_feature = (HashMap<String, Object>) feature.getValue();
            List<String> current_feature_loinc = (List<String>) current_feature.get("loinc");
            String current_feature_aggregation = (String) current_feature.get("aggregation");

            ListIterator<Map<String, Object>> data_iterator = patient_data_points.listIterator();
            float current_value = Float.NaN;

            while (data_iterator.hasNext()) {

                // Get properties of current data point
                Map<String, Object> data_point = data_iterator.next();
                String current_data_loinc = (String) data_point.get("loinc_code");
                String current_data_aggregation = (String) data_point.get("aggregation");

                // Check if it matches the properties of the current feature
                if (current_feature_loinc.contains(current_data_loinc)
                        && current_feature_aggregation.equals(current_data_aggregation)) {

                    // Ensure that the data point is not missing, in that case it should remain NaN
                    if (data_point.get("value") != null) {
                        current_value = ((Double) data_point.get("value")).floatValue();
                    }

                    // Stop iterating through the data points if a matching one has been found
                    break;
                }
            }
            patient_feature_values[feature_index] = current_value;
        }
        return patient_feature_values;
    }

    public float get_patient_weight(List<Map<String, Object>> patient_data_points) {

        /**
         * Extract the patient's weight from the data points.
         * This is required because the weight is not directly used as a model input,
         * but necessary for normalizing the urine output.
         * 
         * @param patient_data_points A list of data points available for a single
         *                            patient as read from file.
         * @return The patient's weight.
         */

        ListIterator<Map<String, Object>> data_iterator = patient_data_points.listIterator();
        float weight = Float.NaN;

        while (data_iterator.hasNext()) {

            Map<String, Object> data_point = data_iterator.next();
            String current_data_loinc = (String) data_point.get("loinc_code");

            if (current_data_loinc.equals(this.urine_output_loinc)) {

                // Ensure that the data point is not missing, then the NaN should be kept
                if (data_point.get("value") != null) {
                    weight = ((Double) data_point.get("value")).floatValue();
                }
                break;
            }
        }
        return weight;
    }

    public Map<String, Map<String, Object>> read_model_features(String data_path) throws java.io.IOException {

        /**
         * Read model features from a JSON file and return them as in map instance.
         * 
         * @param data_path The path to the JSON file containing the model features.
         * @return A map instance containing the model features.
         */

        ObjectMapper mapper = new ObjectMapper();
        Map<String, Map<String, Object>> map = mapper.readValue(Paths.get(data_path).toFile(), Map.class);

        return map;
    }

    private Map<String, List<Map<String, Object>>> read_data_file(String data_path) throws java.io.IOException {

        /**
         * Read patient data from a JSON file and return it as a map instance.
         * 
         * @param data_path Path to the JSON file containing the patient data.
         * @return Map instance containing the patient data.
         */

        ObjectMapper mapper = new ObjectMapper();
        Map<String, List<Map<String, Object>>> map = mapper.readValue(Paths.get(data_path).toFile(), Map.class);

        return map;
    }
}
