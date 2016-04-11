--------------------------------------------------------------------------------
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE OverloadedStrings #-}

import qualified Data.Char                       as C
import           Data.List
import           Data.Monoid                     (mappend)
import qualified Data.Set                        as S
import           Hakyll
import           System.Environment
import           System.FilePath.Posix
import           Text.Blaze.Html                 (toHtml, toValue, (!))
import           Text.Blaze.Html.Renderer.String (renderHtml)
import qualified Text.Blaze.Html5                as H
import qualified Text.Blaze.Html5.Attributes     as A
import           Text.Pandoc.Options


--------------------------------------------------------------------------------
main :: IO ()
main = do
    (action:_) <- getArgs
    let postsPattern = if action == "watch"
                       then "posts/**.md" .||. "drafts/**.md"
                       else "posts/**.md"

    hakyll $ do

        pages <- buildPages postsPattern

        categories <- buildCategories postsPattern (fromCapture "*/index.html")

        match "images/*" $ do
            route   idRoute
            compile copyFileCompiler

        -- Tell hakyll to watch the less files
        match "css/*.less" $ compile getResourceBody

        -- Compile the main less file
        -- We tell hakyll it depends on all the less files,
        -- so it will recompile it when needed
        d <- makePatternDependency "css/*.less"
        rulesExtraDependencies [d] $ create ["css/main.css"] $ do
            route idRoute
            compile $ loadBody "css/main.less"
                >>= makeItem
                >>= withItemBody
                  (unixFilter "lessc" ["--clean-css","--include-path=css","-"])

        match "pages/*.md" $ do
            route   niceRoutePages
            compile $ pandocMathCompiler
                >>= loadAndApplyTemplate "templates/default.html" defaultContext
                >>= removeIndexHtml
                >>= relativizeUrls

        match postsPattern $ do
            route niceRoute
            compile $ pandocMathCompiler
                >>= saveSnapshot "content"
                >>= loadAndApplyTemplate "templates/post.html"    (postCtxWithCat categories)
                >>= loadAndApplyTemplate "templates/default.html" (postCtxWithCat categories)
                >>= removeIndexHtml
                >>= relativizeUrls

        create ["archive.html"] $ do
            route idRoute
            compile $ do
                posts <- recentFirst =<< loadAll postsPattern
                let archiveCtx =
                        listField "posts" (postCtxWithCat categories) (return posts) `mappend`
                        constField "title" "Archives"            `mappend`
                        defaultContext

                makeItem ""
                    >>= loadAndApplyTemplate "templates/archive.html" archiveCtx
                    >>= loadAndApplyTemplate "templates/default.html" archiveCtx
                    >>= removeIndexHtml
                    >>= relativizeUrls

        match "404.md" $ do
          route niceRoute
          compile $ do
            let notFoundCtx =
                    constField "title" "404 Page Not Found" `mappend`
                    defaultContext

            pandocMathCompiler
                    >>= loadAndApplyTemplate "templates/default.html" notFoundCtx
                    >>= removeIndexHtml
                    >>= relativizeUrls

        paginateRules pages $ \index pattern -> do
          route idRoute
          compile $ do
            let posts = recentFirst =<< loadAllSnapshots postsPattern "content"
            let indexCtx =
                  listField "posts" (previewContextWithCat categories) (takeFromTo start end <$> posts) `mappend`
                  constField "title" "Home" `mappend`
                  paginateContext pages index `mappend`
                  defaultContext
                  where
                    start = 5 * (index - 1)
                    end = 5 * index

            makeItem ""
              >>= loadAndApplyTemplate "templates/posts.html" indexCtx
              >>= loadAndApplyTemplate "templates/default.html" indexCtx
              >>= removeIndexHtml
              >>= relativizeUrls

        match "templates/*" $ compile templateCompiler

        let buildPaginatedTag (tag, idlist) = do
                pages <- buildTagPages tag idlist
                paginateRules pages $ \index pattern -> do
                  route idRoute
                  compile $ do
                    let posts = recentFirst =<< loadAllSnapshots (fromList idlist) "content"
                    let indexCtx =
                          listField "posts" (previewContextWithCat categories) (takeFromTo start end <$> posts) `mappend`
                          constField "title" "Home" `mappend`
                          paginateContext pages index `mappend`
                          defaultContext
                          where
                            start = 5 * (index - 1)
                            end = 5 * index

                    makeItem ""
                      >>= loadAndApplyTemplate "templates/posts.html" indexCtx
                      >>= loadAndApplyTemplate "templates/default.html" indexCtx
                      >>= removeIndexHtml
                      >>= relativizeUrls

        mapM_ buildPaginatedTag $ tagsMap categories

--------------------------------------------------------------------------------
postCtx :: Context String
postCtx =
    dateField "date" "%B %e, %Y" `mappend`
    defaultContext

postCtxWithCat :: Tags -> Context String
postCtxWithCat categories = categoryField' "category" categories `mappend` postCtx

previewContext :: Context String
previewContext = teaserField "preview" "content" `mappend` postCtx

previewContextWithCat :: Tags -> Context String
previewContextWithCat categories = teaserField "preview" "content" `mappend` postCtxWithCat categories

categoryField' =
  tagsFieldWith getCategory simpleRenderLink' (mconcat . intersperse ", ")

-- | Obtain categories from a page.
getCategory :: MonadMetadata m => Identifier -> m [String]
getCategory = return . return . takeBaseName . takeDirectory . toFilePath

simpleRenderLink' :: String -> Maybe FilePath -> Maybe H.Html
simpleRenderLink' _   Nothing         = Nothing
simpleRenderLink' tag (Just filePath) =
  Just $ H.a ! A.href (toValue $ toUrl filePath) $ toHtml (firstUpper tag)

firstUpper :: String -> String
firstUpper (hd:tl) = C.toUpper hd : tl
firstUpper [] = []


pandocMathCompiler =
    let mathExtensions = [Ext_tex_math_dollars, Ext_tex_math_double_backslash,
                          Ext_latex_macros]
        defaultExtensions = writerExtensions defaultHakyllWriterOptions
        newExtensions = foldr S.insert defaultExtensions mathExtensions
        writerOptions = defaultHakyllWriterOptions {
                          writerExtensions = newExtensions,
                          writerHTMLMathMethod = MathJax ""
                        }
    in pandocCompilerWith defaultHakyllReaderOptions writerOptions

-- replace a foo/bar.md by foo/bar/index.html
-- this way the url looks like: foo/bar in most browsers
niceRoute :: Routes
niceRoute = customRoute createIndexRoute
    where
      createIndexRoute ident =
        takeDirectory p </> takeBaseName p </> "index.html"
        where p = toFilePath ident

niceRoutePages :: Routes
niceRoutePages = customRoute createIndexRoute
    where
      createIndexRoute ident =
        takeBaseName p </> "index.html"
        where p = toFilePath ident

-- replace url of the form foo/bar/index.html by foo/bar
removeIndexHtml :: Item String -> Compiler (Item String)
removeIndexHtml item = return $ fmap (withUrls removeIndexStr) item
  where
    removeIndexStr :: String -> String
    removeIndexStr url = case splitFileName url of
        (dir, "index.html") | isLocal dir -> dir
        _                                 -> url
        where isLocal uri = not ("://" `isInfixOf` uri)

buildPages :: (MonadMetadata m) => Pattern -> m Paginate
buildPages pattern =
  buildPaginateWith
    (return . paginateEvery 5)
    pattern
    (\index ->
      if index == 1
        then fromFilePath "index.html"
        else fromFilePath $ "page-" ++ show index ++ "/index.html")

buildTagPages :: (MonadMetadata m) => String -> [Identifier] -> m Paginate
buildTagPages tag identifiers =
  buildPaginateWith
    (return . paginateEvery 5)
    (fromList identifiers)
    (\index ->
      if index == 1
        then fromFilePath $ tag ++ "/index.html"
        else fromFilePath $ tag ++ "/page-" ++ show index ++ "/index.html")

takeFromTo :: Int -> Int -> [a] -> [a]
takeFromTo start end = drop start . take end
