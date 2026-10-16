pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // AndroidMath (LaTeX rendering) is published only through JitPack; nothing else may resolve from it.
        maven("https://jitpack.io") { content { includeGroup("com.github.gregcockroft") } }
    }
}
rootProject.name = "Equity"
include(":app")
