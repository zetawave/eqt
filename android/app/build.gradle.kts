import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Release signing reads an untracked keystore.properties (see scripts/android-app.ps1); without it the
// release APK is built unsigned.
val signing = Properties().apply {
    rootProject.file("keystore.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
}

android {
    namespace = "org.equity.app"
    compileSdk = 36
    ndkVersion = "28.2.13676358"
    defaultConfig {
        applicationId = "org.equity.app"
        minSdk = 28
        targetSdk = 36
        versionCode = 2
        versionName = "0.2.0"
        testInstrumentationRunner = "org.equity.app.NativeSmokeTest"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake {
            arguments += listOf("-DANDROID_STL=c++_shared", "-DCMAKE_BUILD_TYPE=Release")
            targets += "eqt_jni"
            // GGML CPU backends are dlopen'ed modules, not link dependencies; list them so they are packaged.
            targets += listOf("8.0_1", "8.2_1", "8.2_2", "8.6_1", "9.0_1", "9.2_1", "9.2_2").map { "ggml-cpu-android_armv$it" }
        } }
    }
    signingConfigs {
        if (signing.containsKey("storeFile")) create("release") {
            storeFile = rootProject.file(signing.getProperty("storeFile"))
            storePassword = signing.getProperty("storePassword")
            keyAlias = signing.getProperty("keyAlias")
            keyPassword = signing.getProperty("keyPassword")
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.findByName("release")
        }
    }
    buildFeatures { compose = true }
    externalNativeBuild { cmake { path = file("../../CMakeLists.txt"); version = "3.22.1" } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    sourceSets["main"].assets.srcDir("../../licenses")
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2025.04.01")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.work:work-runtime-ktx:2.10.0")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
